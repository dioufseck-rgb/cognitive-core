"""Offline tests of search, source reading, and credential/network boundaries."""

from __future__ import annotations

import gzip
import importlib.util
import json
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

try:
    import httpx
except ImportError:
    httpx = None

from cognitive_core.engine.tools import ToolRegistry
from cognitive_core.inquiry.research import ResearchError, SerperResearchProvider


PUBLIC_IP = "93.184.216.34"
FIXTURE_KEY = "fixture-serper-secret"


def _small_pdf(pages: list[str]) -> bytes:
    """A two-font-free fixture PDF, generated without a test-only dependency."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{' '.join(f'{4 + 2 * i} 0 R' for i in range(len(pages)))}] /Count {len(pages)} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for index, text in enumerate(pages):
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * index} 0 R >>".encode()
        )
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
    data = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return bytes(data)


@unittest.skipIf(httpx is None, "httpx is an optional inquiry dependency")
class TestInquiryResearch(unittest.TestCase):
    def provider(self, handler, **kwargs):
        client = httpx.Client(
            transport=httpx.MockTransport(handler),
            # These must NOT be inherited by a source request.
            headers={"Authorization": "Bearer inherited-secret", "X-API-KEY": "inherited-key"},
            cookies={"session": "inherited-cookie"},
            auth=("inherited-user", "inherited-password"),
        )
        self.addCleanup(client.close)
        return SerperResearchProvider(
            api_key=kwargs.pop("api_key", FIXTURE_KEY),
            client=client,
            resolver=kwargs.pop("resolver", lambda host, port: [PUBLIC_IP]),
            **kwargs,
        )

    def test_search_sends_explicit_query_and_normalizes_hits(self):
        requests = []

        def handler(request):
            requests.append(request)
            self.assertEqual(str(request.url), "https://google.serper.dev/search")
            self.assertEqual(request.headers["X-API-KEY"], FIXTURE_KEY)
            self.assertNotIn("Authorization", request.headers)
            self.assertNotIn("Cookie", request.headers)
            self.assertEqual(json.loads(request.content), {"q": "research question", "num": 2})
            return httpx.Response(200, json={"organic": [
                {"title": "  First\npage ", "link": "https://example.org/a#part", "snippet": "Some\n evidence"},
                {"title": "Duplicate", "link": "https://example.org/a#other"},
                {"title": "Not a webpage", "link": "file:///etc/passwd"},
                {"title": "Second", "link": "https://example.org/b", "snippet": "More evidence"},
            ]})

        hits = self.provider(handler).search(" research question ", num=2)
        self.assertEqual(len(requests), 1)
        self.assertEqual(hits, [
            {"title": "First page", "url": "https://example.org/a", "snippet": "Some evidence"},
            {"title": "Second", "url": "https://example.org/b", "snippet": "More evidence"},
        ])

    def test_key_is_read_lazily_from_environment(self):
        def handler(request):
            self.assertEqual(request.headers["X-API-KEY"], "late-fixture-key")
            return httpx.Response(200, json={"organic": []})

        provider = self.provider(handler, api_key=None)
        with patch.dict("os.environ", {"SERPER_API_KEY": "late-fixture-key"}):
            self.assertEqual(provider.search("question"), [])

    def test_missing_key_fails_before_network(self):
        provider = self.provider(lambda request: self.fail("Unexpected request"), api_key="")
        with self.assertRaisesRegex(ResearchError, "Set SERPER_API_KEY"):
            provider.search("question")

    def test_invalid_query_and_result_count_fail_before_network(self):
        provider = self.provider(lambda request: self.fail("Unexpected request"))
        for query in (None, "", "  ", "q" * 2001):
            with self.subTest(query_length=len(query) if query is not None else None):
                with self.assertRaises(ResearchError):
                    provider.search(query)
        for count in (0, 11, True, "5"):
            with self.subTest(count=count):
                with self.assertRaises(ResearchError):
                    provider.search("question", num=count)

    def test_api_errors_never_echo_response_or_secret(self):
        for status, expected in ((401, "rejected"), (403, "rejected"), (402, "quota"), (429, "quota"), (503, "HTTP 503")):
            with self.subTest(status=status):
                provider = self.provider(lambda request: httpx.Response(status, text=FIXTURE_KEY))
                with self.assertRaisesRegex(ResearchError, expected) as error:
                    provider.search("question")
                self.assertNotIn(FIXTURE_KEY, str(error.exception))

    def test_transport_errors_never_echo_request_credentials(self):
        def handler(request):
            raise httpx.ConnectError(f"request failed: {FIXTURE_KEY}", request=request)

        with self.assertRaisesRegex(ResearchError, "Search request failed") as error:
            self.provider(handler).search("question")
        self.assertNotIn(FIXTURE_KEY, str(error.exception))

    def test_invalid_search_json_and_shape_are_explicit_failures(self):
        for content in (b"not JSON", b'[]', b'{"organic": {}}', b'{"error": "upstream secret"}'):
            with self.subTest(content=content):
                provider = self.provider(lambda request: httpx.Response(200, content=content))
                with self.assertRaises(ResearchError) as error:
                    provider.search("question")
                self.assertNotIn("upstream secret", str(error.exception))

    def test_source_is_pinned_and_does_not_inherit_credentials(self):
        def handler(request):
            self.assertEqual(str(request.url), f"https://{PUBLIC_IP}/paper?q=1")
            self.assertEqual(request.headers["Host"], "example.org")
            self.assertEqual(request.extensions["sni_hostname"], "example.org")
            for header in ("Authorization", "X-API-KEY", "Cookie"):
                self.assertNotIn(header, request.headers)
            return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text="""
                <html><head><title>A useful paper</title><script>secretScript()</script></head>
                <body><nav>Menu links</nav><main><h1>Finding</h1>
                <p>Evidence <strong>supports</strong> a qualified conclusion.</p>
                <style>.hidden{}</style><footer>Footer boilerplate</footer></main></body></html>
            """)

        source = self.provider(handler).read("https://example.org/paper?q=1#results")
        self.assertEqual(source["url"], "https://example.org/paper?q=1")
        self.assertEqual(source["requested_url"], source["url"])
        self.assertEqual(source["title"], "A useful paper")
        self.assertIn("Evidence supports a qualified conclusion.", source["text"])
        for hidden in ("secretScript", "Menu links", "Footer boilerplate", ".hidden"):
            self.assertNotIn(hidden, source["text"])
        self.assertFalse(source["truncated"])
        self.assertTrue(source["retrieved_at"].endswith("+00:00"))

    def test_source_read_does_not_require_serper_key(self):
        provider = self.provider(
            lambda request: httpx.Response(200, headers={"content-type": "text/plain"}, text="Public source"),
            api_key="",
        )
        self.assertEqual(provider.read("https://example.org/")["text"], "Public source")

    def test_private_unsafe_and_credential_urls_are_rejected(self):
        provider = self.provider(lambda request: self.fail("Unexpected request"))
        urls = [
            "file:///etc/passwd", "ftp://example.org/file", "http://localhost/", "http://127.0.0.1/",
            "http://10.0.0.1/", "http://169.254.169.254/latest/meta-data/", "http://[::1]/",
            "http://[::ffff:127.0.0.1]/", "http://[ff02::1]/", "https://user:password@example.org/",
            "https://example.org:8443/", "http://example.org:443/", "https://example.org:0/",
            "https://example.org/\nheader", "https://example.org\\@127.0.0.1/", "http://host.local/",
        ]
        for url in urls:
            with self.subTest(url=url):
                with self.assertRaises(ResearchError):
                    provider.read(url)

    def test_dns_results_must_all_be_public(self):
        for addresses in (["127.0.0.1"], ["169.254.169.254"], [PUBLIC_IP, "10.0.0.1"], [], ["not-an-address"]):
            with self.subTest(addresses=addresses):
                provider = self.provider(lambda request: self.fail("Unexpected request"), resolver=lambda host, port: addresses)
                with self.assertRaises(ResearchError):
                    provider.read("https://example.org/source")

    def test_dns_failure_is_sanitized(self):
        def resolver(host, port):
            raise OSError(FIXTURE_KEY)

        provider = self.provider(lambda request: self.fail("Unexpected request"), resolver=resolver)
        with self.assertRaisesRegex(ResearchError, "Unable to resolve") as error:
            provider.read("https://example.org/source")
        self.assertNotIn(FIXTURE_KEY, str(error.exception))

    def test_dns_timeout_returns_without_waiting_for_resolver(self):
        def resolver(host, port):
            time.sleep(0.15)
            return [PUBLIC_IP]

        provider = self.provider(lambda request: self.fail("Unexpected request"), resolver=resolver, timeout_seconds=0.01)
        started = time.monotonic()
        with self.assertRaisesRegex(ResearchError, "lookup exceeded"):
            provider.read("https://example.org/source")
        self.assertLess(time.monotonic() - started, 0.12)

    def test_public_redirect_is_resolved_again_and_preserves_final_url(self):
        requests = []
        lookups = []

        def resolver(host, port):
            lookups.append(host)
            return [PUBLIC_IP if host == "example.org" else "8.8.8.8"]

        def handler(request):
            requests.append(request)
            if request.headers["Host"] == "example.org":
                return httpx.Response(302, headers={"location": "https://other.example/report"})
            return httpx.Response(200, headers={"content-type": "text/plain"}, text="Redirected evidence")

        source = self.provider(handler, resolver=resolver).read("https://example.org/start")
        self.assertEqual(lookups, ["example.org", "other.example"])
        self.assertEqual(str(requests[1].url), "https://8.8.8.8/report")
        self.assertEqual(source["url"], "https://other.example/report")
        self.assertEqual(source["requested_url"], "https://example.org/start")

    def test_private_redirect_is_blocked_before_second_request(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(302, headers={"location": "http://169.254.169.254/secret"})

        with self.assertRaises(ResearchError):
            self.provider(handler).read("https://example.org/start")
        self.assertEqual(len(requests), 1)

    def test_redirect_to_private_dns_is_blocked(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(302, headers={"location": "https://other.example/"})

        resolver = lambda host, port: [PUBLIC_IP if host == "example.org" else "127.0.0.1"]
        with self.assertRaisesRegex(ResearchError, "non-public"):
            self.provider(handler, resolver=resolver).read("https://example.org/start")
        self.assertEqual(len(requests), 1)

    def test_redirect_loop_and_hop_limit_are_explicit(self):
        provider = self.provider(lambda request: httpx.Response(302, headers={"location": "/start"}))
        with self.assertRaisesRegex(ResearchError, "loop"):
            provider.read("https://example.org/start")
        provider = self.provider(lambda request: httpx.Response(302, headers={"location": "/next"}), max_redirects=0)
        with self.assertRaisesRegex(ResearchError, "redirect limit"):
            provider.read("https://example.org/start")

    def test_long_text_is_explicitly_marked_truncated(self):
        provider = self.provider(
            lambda request: httpx.Response(200, headers={"content-type": "text/plain"}, text="evidence " * 20),
            max_source_chars=25,
        )
        source = provider.read("https://example.org/")
        self.assertTrue(source["truncated"])
        self.assertEqual(len(source["text"]), 25)

    def test_declared_and_streamed_oversize_bodies_are_rejected(self):
        for headers in ({"content-length": "101"}, {}):
            with self.subTest(headers=headers):
                provider = self.provider(
                    lambda request: httpx.Response(200, headers={"content-type": "text/plain", **headers}, stream=httpx.ByteStream(b"x" * 101)),
                    max_response_bytes=100,
                )
                with self.assertRaisesRegex(ResearchError, "size limit"):
                    provider.read("https://example.org/")

    def test_compressed_source_is_decoded_with_a_separate_size_limit(self):
        def response(text):
            return httpx.Response(
                200, headers={"content-type": "text/plain", "content-encoding": "gzip"},
                stream=httpx.ByteStream(gzip.compress(text)),
            )

        provider = self.provider(lambda request: response(b"Small source"), max_response_bytes=100)
        self.assertEqual(provider.read("https://example.org/")["text"], "Small source")
        provider = self.provider(lambda request: response(b"x" * 10000), max_response_bytes=100)
        with self.assertRaisesRegex(ResearchError, "size limit"):
            provider.read("https://example.org/")

    def test_incomplete_compression_is_a_failure(self):
        body = gzip.compress(b"A useful source")[:-5]
        provider = self.provider(lambda request: httpx.Response(
            200, headers={"content-type": "text/plain", "content-encoding": "gzip"},
            stream=httpx.ByteStream(body),
        ))
        with self.assertRaisesRegex(ResearchError, "incomplete compressed"):
            provider.read("https://example.org/")

    def test_empty_unsupported_and_http_error_sources_are_explicit(self):
        for response, expected in (
            (httpx.Response(200, headers={"content-type": "text/html"}, text="<script>only script</script>"), "no readable text"),
            (httpx.Response(200, headers={"content-type": "image/png"}, content=b"binary"), "unsupported content type"),
            (httpx.Response(403, text=FIXTURE_KEY), "HTTP 403"),
        ):
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ResearchError, expected) as error:
                    self.provider(lambda request: response).read("https://example.org/")
                self.assertNotIn(FIXTURE_KEY, str(error.exception))

    def test_text_encoding_is_respected_and_unknown_encoding_falls_back(self):
        provider = self.provider(lambda request: httpx.Response(
            200, headers={"content-type": "text/plain; charset=iso-8859-1"}, content="café".encode("latin-1"),
        ))
        self.assertEqual(provider.read("https://example.org/")["text"], "café")
        provider = self.provider(lambda request: httpx.Response(
            200, headers={"content-type": "text/plain; charset=unknown-encoding"}, content=b"Text",
        ))
        self.assertEqual(provider.read("https://example.org/")["text"], "Text")

    def test_pdf_missing_decoder_is_an_explicit_failure(self):
        provider = self.provider(lambda request: httpx.Response(
            200, headers={"content-type": "application/pdf"}, content=b"%PDF-test",
        ))
        with patch.dict(sys.modules, {"pdfplumber": None}):
            with self.assertRaisesRegex(ResearchError, "optional pdfplumber"):
                provider.read("https://example.org/paper.pdf")

    @unittest.skipUnless(importlib.util.find_spec("pdfplumber"), "optional PDF decoder unavailable")
    def test_pdf_page_limit_marks_result_truncated(self):
        provider = self.provider(lambda request: httpx.Response(
            200, headers={"content-type": "application/pdf"}, content=_small_pdf(["First page", "Second page"]),
        ), max_pdf_pages=1)
        source = provider.read("https://example.org/paper.pdf")
        self.assertEqual(source["text"], "First page")
        self.assertTrue(source["truncated"])

    def test_pdf_worker_has_no_credentials_and_honors_timeout(self):
        provider = self.provider(lambda request: httpx.Response(
            200, headers={"content-type": "application/pdf"}, content=b"%PDF-test",
        ))
        with patch("cognitive_core.inquiry.research.importlib.util.find_spec", return_value=object()):
            with patch("cognitive_core.inquiry.research.subprocess.run", side_effect=subprocess.TimeoutExpired("fixture", 1)) as worker:
                with patch.dict("os.environ", {"SERPER_API_KEY": FIXTURE_KEY, "OPENAI_API_KEY": "another-fixture-secret"}):
                    with self.assertRaisesRegex(ResearchError, "PDF extraction exceeded"):
                        provider.read("https://example.org/paper.pdf")
                kwargs = worker.call_args.kwargs
                self.assertNotIn("SERPER_API_KEY", kwargs["env"])
                self.assertNotIn("OPENAI_API_KEY", kwargs["env"])
                self.assertGreater(kwargs["timeout"], 0)
                self.assertLessEqual(kwargs["timeout"], provider.timeout_seconds)

    def test_tool_registry_accepts_explicit_arguments_and_records_failure(self):
        calls = []

        def handler(request):
            calls.append(request)
            if request.method == "POST":
                return httpx.Response(200, json={"organic": [{"title": "A source", "link": "https://example.org/paper", "snippet": "An unread snippet"}]})
            return httpx.Response(200, headers={"content-type": "text/plain"}, text="Read source content")

        registry = ToolRegistry()
        self.provider(handler).register_tools(registry)
        search = registry.call("web_search", {"query": "question", "num": 3})
        self.assertEqual(search.status, "success")
        self.assertEqual(len(search.data["results"]), 1)
        read = registry.call("read_source", {"url": search.data["results"][0]["url"]})
        self.assertEqual(read.status, "success")
        self.assertEqual(read.data["text"], "Read source content")
        failed = registry.call("read_source", {"url": "http://127.0.0.1/"})
        self.assertEqual(failed.status, "failed")
        self.assertIsNone(failed.data)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
