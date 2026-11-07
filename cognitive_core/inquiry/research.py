"""Small, explicit web tools for reasoning inquiries.

Search discovers URLs; ``read_source`` returns source text for examination. The
adapter never treats a search snippet as a read source, and never passes the
Serper credential to a source website. Returned text is untrusted evidence, not
instructions for the inquiry controller.
"""

from __future__ import annotations

import codecs
import importlib.util
import ipaddress
import json
import os
import re
import socket
import subprocess
import sys
import time
import zlib
from concurrent.futures import Future, TimeoutError as FutureTimeout
from contextlib import nullcontext
from datetime import datetime, timezone
from email.message import Message
from html.parser import HTMLParser
from threading import Thread
from typing import Any, Callable, Iterable, TYPE_CHECKING
from urllib.parse import urljoin, urlsplit, urlunsplit

if TYPE_CHECKING:
    import httpx

    from cognitive_core.engine.tools import ToolRegistry


class ResearchError(RuntimeError):
    """A safe, user-facing failure without provider response bodies or secrets."""


def _httpx():
    try:
        import httpx
    except ImportError:
        raise ResearchError(
            "Web research requires httpx. Install cognitive-core[inquiry]."
        ) from None
    return httpx


def _resolve_addresses(host: str, port: int) -> list[str]:
    return list(dict.fromkeys(
        item[4][0] for item in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    ))


def _normalize_url(url: str) -> tuple[str, str, int]:
    if not isinstance(url, str) or not url or len(url) > 8192:
        raise ResearchError("Source URL must be a nonempty HTTP or HTTPS URL.")
    if any(ord(char) <= 32 or ord(char) == 127 for char in url) or "\\" in url:
        raise ResearchError("Source URL contains invalid characters.")
    try:
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            raise ValueError
        if parsed.username is not None or parsed.password is not None:
            raise ValueError
        host = parsed.hostname.rstrip(".").encode("idna").decode("ascii").lower()
        port = parsed.port if parsed.port is not None else (443 if parsed.scheme.lower() == "https" else 80)
        if port != (443 if parsed.scheme.lower() == "https" else 80):
            raise ValueError
        if "%" in host or not host:
            raise ValueError
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            if (
                "." not in host
                or host.endswith((".localhost", ".local", ".internal"))
                or not re.fullmatch(r"[a-z0-9.-]+", host)
            ):
                raise ValueError from None
        else:
            if not _is_public_address(address):
                raise ValueError
        authority = f"[{host}]" if ":" in host else host
        normalized = urlunsplit((parsed.scheme.lower(), authority, parsed.path or "/", parsed.query, ""))
        return normalized, host, port
    except (ValueError, UnicodeError):
        raise ResearchError("Only public HTTP/HTTPS URLs on standard ports are allowed.") from None


def _is_public_address(address: Any) -> bool:
    # IPv4-mapped addresses must satisfy the IPv4 policy as well.
    mapped = getattr(address, "ipv4_mapped", None)
    return bool(
        address.is_global
        and not address.is_multicast
        and not address.is_unspecified
        and not address.is_reserved
        and (mapped is None or _is_public_address(mapped))
    )


class _PageText(HTMLParser):
    """Extract readable text without running scripts or following links."""

    _hidden = {"script", "style", "nav", "header", "footer", "form", "svg", "noscript", "template", "iframe", "head"}
    _blocks = {"p", "div", "section", "article", "main", "li", "ul", "ol", "table", "tr", "td", "th", "br", "hr", "pre", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self.hidden: list[str] = []
        self.in_title = False

    def handle_starttag(self, tag: str, attrs):
        if tag == "title":
            self.in_title = True
        if tag in self._hidden:
            self.hidden.append(tag)
        if not self.hidden and tag in self._blocks:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str):
        if tag == "title":
            self.in_title = False
        if tag in self.hidden:
            index = len(self.hidden) - 1 - self.hidden[::-1].index(tag)
            del self.hidden[index:]
        if not self.hidden and tag in self._blocks:
            self.parts.append("\n")

    def handle_data(self, data: str):
        if self.in_title:
            self.title_parts.append(data)
        elif not self.hidden:
            self.parts.append(data)

    def result(self) -> tuple[str, str]:
        title = " ".join(" ".join(self.title_parts).split())[:512]
        lines = [" ".join(line.split()) for line in "".join(self.parts).splitlines()]
        return title, "\n".join(line for line in lines if line)


_PDF_WORKER = """
import json, math, sys
from io import BytesIO
max_chars, max_pages = int(sys.argv[1]), int(sys.argv[2])
try:
    import resource
    cpu_limit = max(1, math.ceil(float(sys.argv[3])))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 1))
    resource.setrlimit(resource.RLIMIT_AS, (768 * 1024 * 1024, 768 * 1024 * 1024))
except (ImportError, OSError, ValueError):
    pass
import pdfplumber
parts, total = [], 0
with pdfplumber.open(BytesIO(sys.stdin.buffer.read())) as pdf:
    count = len(pdf.pages)
    for page in pdf.pages[:max_pages]:
        text = page.extract_text() or ''
        parts.append(text[:max_chars + 1 - total])
        total += len(parts[-1]) + 1
        if total > max_chars:
            break
print(json.dumps({'text': '\\n'.join(parts), 'truncated': count > len(parts)}))
"""


class SerperResearchProvider:
    """Serper discovery and bounded, public source reading.

    ``client`` and ``resolver`` allow offline tests. The default client has no
    environment-derived proxy or credentials. Read requests use a validated IP
    with the original Host and TLS server name, avoiding a second DNS lookup.
    An injected client is caller-owned and must preserve HTTPX's TLS behavior.
    """

    SEARCH_URL = "https://google.serper.dev/search"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout_seconds: float = 20.0,
        max_source_chars: int = 30000,
        max_response_bytes: int = 2_000_000,
        max_redirects: int = 4,
        max_pdf_pages: int = 30,
        client: httpx.Client | None = None,
        resolver: Callable[[str, int], Iterable[str]] | None = None,
    ):
        if timeout_seconds <= 0 or max_source_chars <= 0 or max_response_bytes <= 0:
            raise ValueError("Research time and size limits must be positive.")
        if max_redirects < 0 or max_pdf_pages <= 0:
            raise ValueError("Redirect/page limits are invalid.")
        self._api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_source_chars = max_source_chars
        self.max_response_bytes = max_response_bytes
        self.max_redirects = max_redirects
        self.max_pdf_pages = max_pdf_pages
        self._client = client
        self._resolver = resolver or _resolve_addresses

    def _client_context(self):
        if self._client is not None:
            return nullcontext(self._client)
        return _httpx().Client(
            timeout=self.timeout_seconds, trust_env=False, follow_redirects=False,
            limits=_httpx().Limits(max_keepalive_connections=0),
        )

    def _request(self, client, method: str, url: str, deadline: float, *, headers=None, payload=None, server_name=None):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ResearchError("Web request exceeded its time limit.")
        httpx = _httpx()
        extensions = {"timeout": {key: remaining for key in ("connect", "read", "write", "pool")}}
        if server_name:
            extensions["sni_hostname"] = server_name
        # Construct a standalone request: no client default headers or cookies.
        request = httpx.Request(
            method, url, headers=headers, json=payload, extensions=extensions,
        )
        return client.send(request, stream=True, auth=None, follow_redirects=False)

    def _body(self, response, deadline: float) -> bytes:
        """Bound both compressed transfer bytes and decompressed bytes."""
        limit = self.max_response_bytes
        length = response.headers.get("content-length", "")
        if length.isdigit() and int(length) > limit:
            raise ResearchError("Web response exceeds the configured size limit.")
        # Mock transports and caching clients may already have consumed a body.
        # HTTPX has decoded it in that case, so enforce the decoded limit here.
        if response.is_stream_consumed:
            body = response.content
            if len(body) > limit:
                raise ResearchError("Web response exceeds the configured size limit.")
            return body
        encoding = response.headers.get("content-encoding", "identity").strip().lower()
        if encoding in {"", "identity"}:
            decoder = None
        elif encoding == "gzip":
            decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        elif encoding == "deflate":
            decoder = zlib.decompressobj(zlib.MAX_WBITS)
        else:
            raise ResearchError("Web response uses an unsupported content encoding.")
        parts = []
        transferred = decoded_size = 0
        try:
            for raw in response.iter_raw():
                if time.monotonic() > deadline:
                    raise ResearchError("Web request exceeded its time limit.")
                transferred += len(raw)
                if transferred > limit:
                    raise ResearchError("Web response exceeds the configured size limit.")
                decoded = decoder.decompress(raw, limit - decoded_size + 1) if decoder else raw
                decoded_size += len(decoded)
                if decoded_size > limit:
                    raise ResearchError("Web response exceeds the configured size limit.")
                if decoder and decoder.unused_data:
                    raise ResearchError("Web response contains an unsupported compression stream.")
                parts.append(decoded)
            if decoder and not decoder.eof:
                raise ResearchError("Web response contains incomplete compressed content.")
        except zlib.error:
            raise ResearchError("Web response contains invalid compressed content.") from None
        return b"".join(parts)

    def search(self, query: str, *, num: int = 5) -> list[dict[str, Any]]:
        """Return search hits; no hit has yet been read or verified."""
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ResearchError("Search requires a nonempty query of at most 2000 characters.")
        if not isinstance(num, int) or isinstance(num, bool) or not 1 <= num <= 10:
            raise ResearchError("Search result count must be an integer from 1 to 10.")
        key = self._api_key if self._api_key is not None else os.environ.get("SERPER_API_KEY", "")
        if not isinstance(key, str) or not key.strip():
            raise ResearchError("Set SERPER_API_KEY to enable web search.")
        if "\r" in key or "\n" in key:
            raise ResearchError("SERPER_API_KEY has an invalid format.")
        deadline = time.monotonic() + self.timeout_seconds
        try:
            with self._client_context() as client:
                response = self._request(
                    client, "POST", self.SEARCH_URL, deadline,
                    headers={"X-API-KEY": key, "Accept": "application/json", "Accept-Encoding": "gzip, deflate"},
                    payload={"q": query.strip(), "num": num},
                )
                try:
                    if response.status_code in {401, 403}:
                        raise ResearchError("Serper rejected the API key or account access.")
                    if response.status_code in {402, 429}:
                        raise ResearchError("Serper quota or rate limit reached; retry after checking the account.")
                    if not 200 <= response.status_code < 300:
                        raise ResearchError(f"Search service returned HTTP {response.status_code}.")
                    data = json.loads(self._body(response, deadline))
                finally:
                    response.close()
        except ResearchError:
            raise
        except Exception:
            # Upstream exceptions may include request headers or credential data.
            raise ResearchError("Search request failed or returned an invalid response.") from None
        if not isinstance(data, dict) or not isinstance(data.get("organic", []), list) or "error" in data:
            raise ResearchError("Search service returned an invalid result format.")
        hits = []
        seen = set()
        for item in data.get("organic", []):
            if not isinstance(item, dict):
                continue
            try:
                url, _, _ = _normalize_url(item.get("link", ""))
            except ResearchError:
                continue
            if url in seen:
                continue
            seen.add(url)
            hits.append({
                "url": url,
                "title": " ".join(str(item.get("title", "")).split())[:512],
                "snippet": " ".join(str(item.get("snippet", "")).split())[:4000],
            })
            if len(hits) >= num:
                break
        return hits

    def _public_target(self, url: str, deadline: float) -> tuple[str, str, str]:
        logical_url, host, port = _normalize_url(url)
        result: Future = Future()

        def resolve():
            try:
                result.set_result(list(self._resolver(host, port)))
            except Exception as error:
                result.set_exception(error)

        # getaddrinfo has no Python timeout argument. A daemon worker lets the
        # caller honor the request deadline even when a DNS resolver stalls.
        Thread(target=resolve, daemon=True).start()
        try:
            addresses = [
                ipaddress.ip_address(address)
                for address in result.result(timeout=max(0, deadline - time.monotonic()))
            ]
        except FutureTimeout:
            raise ResearchError("Source address lookup exceeded its time limit.") from None
        except Exception:
            raise ResearchError("Unable to resolve the source's public address.") from None
        if not addresses or any(not _is_public_address(address) for address in addresses):
            raise ResearchError("Source resolves to a non-public address and cannot be read.")
        # Prefer IPv4 where both families are available; retain a pinned address.
        address = sorted(addresses, key=lambda item: item.version)[0]
        parsed = urlsplit(logical_url)
        authority = f"[{address}]" if address.version == 6 else str(address)
        target_url = urlunsplit((parsed.scheme, authority, parsed.path, parsed.query, ""))
        return logical_url, host, target_url

    def _extract_pdf(self, body: bytes, deadline: float) -> tuple[str, bool]:
        if importlib.util.find_spec("pdfplumber") is None:
            raise ResearchError("Reading PDF sources requires the optional pdfplumber package.") from None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ResearchError("PDF extraction exceeded its time limit.")
        try:
            # Parser calls can stall within a single page. A subprocess supplies
            # a wall deadline and avoids giving the parser provider credentials.
            environment = {"PATH": os.defpath}
            if "SYSTEMROOT" in os.environ:  # Needed to launch Python on Windows.
                environment["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
            result = subprocess.run(
                [sys.executable, "-I", "-c", _PDF_WORKER,
                 str(self.max_source_chars), str(self.max_pdf_pages), str(remaining)],
                input=body, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                timeout=remaining, check=False, env=environment,
            )
            if result.returncode:
                raise ResearchError("PDF source could not be decoded within the configured limits.")
            data = json.loads(result.stdout)
            if not isinstance(data.get("text"), str) or not isinstance(data.get("truncated"), bool):
                raise ValueError
            return data["text"], data["truncated"]
        except subprocess.TimeoutExpired:
            raise ResearchError("PDF extraction exceeded its time limit.") from None
        except ResearchError:
            raise
        except Exception:
            raise ResearchError("PDF source could not be decoded.") from None

    def read(self, url: str) -> dict[str, Any]:
        """Read one public HTML, plain-text, Markdown, JSON, or PDF source."""
        _httpx()
        deadline = time.monotonic() + self.timeout_seconds
        requested_url, _, _ = _normalize_url(url)
        current = requested_url
        visited = set()
        try:
            with self._client_context() as client:
                for hop in range(self.max_redirects + 1):
                    logical_url, host, target_url = self._public_target(current, deadline)
                    if logical_url in visited:
                        raise ResearchError("Source redirected in a loop.")
                    visited.add(logical_url)
                    host_header = f"[{host}]" if ":" in host else host
                    response = self._request(
                        client, "GET", target_url, deadline, server_name=host,
                        headers={
                            "Host": host_header,
                            "User-Agent": "CognitiveCoreInquiry/0.1 (source reader)",
                            "Accept": "text/html, application/xhtml+xml, text/plain, text/markdown, application/pdf, application/json",
                            "Accept-Encoding": "gzip, deflate",
                            "Connection": "close",
                        },
                    )
                    try:
                        if response.status_code in {301, 302, 303, 307, 308}:
                            if hop >= self.max_redirects:
                                raise ResearchError("Source exceeded the redirect limit.")
                            location = response.headers.get("location")
                            if not location:
                                raise ResearchError("Source returned a redirect without a location.")
                            current = urljoin(logical_url, location)
                            continue
                        if not 200 <= response.status_code < 300:
                            raise ResearchError(f"Source returned HTTP {response.status_code}.")
                        content_type_header = response.headers.get("content-type", "")
                        content_type = content_type_header.split(";", 1)[0].strip().lower()
                        if content_type not in {
                            "text/html", "application/xhtml+xml", "text/plain", "text/markdown",
                            "text/x-markdown", "application/json", "application/pdf",
                        }:
                            raise ResearchError("Source has an unsupported content type.")
                        body = self._body(response, deadline)
                    finally:
                        response.close()
                    break
        except ResearchError:
            raise
        except Exception:
            raise ResearchError("Source request failed or exceeded its time limit.") from None

        title = host
        truncated = False
        if content_type == "application/pdf":
            text, truncated = self._extract_pdf(body, deadline)
        else:
            message = Message()
            message["content-type"] = content_type_header
            encoding = message.get_content_charset() or "utf-8"
            try:
                codecs.lookup(encoding)
            except LookupError:
                encoding = "utf-8"
            decoded = body.decode(encoding, errors="replace").replace("\x00", "")
            if content_type in {"text/html", "application/xhtml+xml"}:
                parser = _PageText()
                parser.feed(decoded)
                title, text = parser.result()
                title = title or host
            else:
                text = decoded.strip()
        if time.monotonic() > deadline:
            raise ResearchError("Source reading exceeded its time limit.")
        if not text.strip():
            raise ResearchError("Source returned no readable text; it may require JavaScript or contain scanned images.")
        truncated = truncated or len(text) > self.max_source_chars
        return {
            "url": logical_url,
            "requested_url": requested_url,
            "title": title,
            "text": text[:self.max_source_chars],
            "truncated": truncated,
            "content_type": content_type,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }

    def register_tools(self, registry: ToolRegistry) -> None:
        """Register explicit-argument tools for a research Retrieve step."""
        registry.register(
            "web_search", self._search_tool,
            description="Search the public web. Arguments: query (string), num (1-10, default 5). Returns unread search hits.",
        )
        registry.register(
            "read_source", self._read_tool,
            description="Read a public source. Argument: url (HTTP/HTTPS). Returns extracted text and provenance.",
        )

    def _search_tool(self, context: dict[str, Any]) -> dict[str, Any]:
        return {"results": self.search(context.get("query", ""), num=context.get("num", 5))}

    def _read_tool(self, context: dict[str, Any]) -> dict[str, Any]:
        return self.read(context.get("url", ""))
