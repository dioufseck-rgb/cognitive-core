"""
Cognitive Core - Production Tool Providers

Three provider types that plug into the ToolRegistry:
  1. API providers — direct service calls (core banking, card processor)
  2. Vector providers — semantic search (prior cases, regulations, narratives)
  3. MCP providers — dynamic tool discovery via Model Context Protocol

Each provider registers tools into a ToolRegistry. The Retrieve primitive
calls them identically — it doesn't know or care about the underlying
transport.
"""

import json
import time
from typing import Any, Callable
from engine.tools import ToolRegistry


# ═══════════════════════════════════════════════════════════════════════
# 1. API PROVIDERS — direct service calls
# ═══════════════════════════════════════════════════════════════════════

class APIProvider:
    """
    Wraps a REST/gRPC service as a set of tools.

    Example usage:
        provider = APIProvider(base_url="https://core-banking.internal")
        provider.add_endpoint(
            name="member_profile",
            path="/v1/members/{member_id}",
            description="Member demographics, tenure, products",
            extract_params=lambda ctx: {"member_id": ctx["member_profile"]["member_id"]},
        )
        provider.register_all(registry)
    """

    def __init__(self, base_url: str, auth_token: str | None = None, timeout_ms: int = 5000):
        self.base_url = base_url
        self.auth_token = auth_token
        self.timeout_ms = timeout_ms
        self._endpoints: list[dict] = []

    def add_endpoint(
        self,
        name: str,
        path: str,
        description: str = "",
        method: str = "GET",
        extract_params: Callable[[dict], dict] | None = None,
        transform: Callable[[dict], dict] | None = None,
        required: bool = False,
        latency_hint_ms: float = 100,
    ):
        """
        Register an API endpoint as a tool.

        Args:
            extract_params: Pulls URL params from the query context.
                           e.g., lambda ctx: {"member_id": ctx["member_id"]}
            transform: Post-processes the API response before returning.
        """
        self._endpoints.append({
            "name": name,
            "path": path,
            "description": description,
            "method": method,
            "extract_params": extract_params or (lambda ctx: {}),
            "transform": transform or (lambda resp: resp),
            "required": required,
            "latency_hint_ms": latency_hint_ms,
        })

    def register_all(self, registry: ToolRegistry):
        """Register all endpoints into a ToolRegistry."""
        for ep in self._endpoints:
            fn = self._make_tool_fn(ep)
            registry.register(
                name=ep["name"],
                fn=fn,
                description=ep["description"],
                latency_hint_ms=ep["latency_hint_ms"],
                required=ep["required"],
            )

    def _make_tool_fn(self, endpoint: dict) -> Callable:
        """Create a tool function that calls the API endpoint."""
        def tool_fn(context: dict[str, Any]) -> dict[str, Any]:
            import httpx  # production dependency

            params = endpoint["extract_params"](context)
            path = endpoint["path"].format(**params)
            url = f"{self.base_url}{path}"

            headers = {"Content-Type": "application/json"}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"

            with httpx.Client(timeout=self.timeout_ms / 1000) as client:
                if endpoint["method"] == "GET":
                    resp = client.get(url, headers=headers)
                else:
                    resp = client.post(url, headers=headers, json=params)

            resp.raise_for_status()
            data = resp.json()
            return endpoint["transform"](data)

        return tool_fn


# ═══════════════════════════════════════════════════════════════════════
# 2. VECTOR PROVIDERS — semantic search
# ═══════════════════════════════════════════════════════════════════════

class VectorProvider:
    """
    Wraps a vector database as a set of semantic search tools.

    Each tool searches a specific collection/namespace with a query
    derived from the workflow context. Returns top-k documents with
    metadata and relevance scores.

    The query formulation is the key design choice:
    - Deterministic: query template filled from context fields
    - Agentic: the LLM formulates the query (handled by Retrieve primitive)

    Example:
        provider = VectorProvider(
            client=chromadb.HttpClient(host="vector-db.internal"),
        )
        provider.add_collection(
            name="prior_disputes",
            collection="dispute_outcomes",
            query_template="card dispute: {dispute_type} merchant: {merchant}",
            extract_query_params=lambda ctx: {
                "dispute_type": ctx.get("_step_classify_dispute_type", {}).get("category", ""),
                "merchant": ctx.get("transaction_detail", {}).get("merchant", ""),
            },
            top_k=5,
        )
        provider.register_all(registry)
    """

    def __init__(self, client: Any = None, embedding_model: str = "text-embedding-004"):
        self.client = client
        self.embedding_model = embedding_model
        self._collections: list[dict] = []

    def add_collection(
        self,
        name: str,
        collection: str,
        description: str = "",
        query_template: str = "",
        extract_query_params: Callable[[dict], dict] | None = None,
        top_k: int = 5,
        score_threshold: float = 0.0,
        metadata_filter: dict | None = None,
        latency_hint_ms: float = 200,
    ):
        self._collections.append({
            "name": name,
            "collection": collection,
            "description": description,
            "query_template": query_template,
            "extract_query_params": extract_query_params or (lambda ctx: {}),
            "top_k": top_k,
            "score_threshold": score_threshold,
            "metadata_filter": metadata_filter,
            "latency_hint_ms": latency_hint_ms,
        })

    def register_all(self, registry: ToolRegistry):
        for col in self._collections:
            fn = self._make_tool_fn(col)
            registry.register(
                name=col["name"],
                fn=fn,
                description=col["description"],
                latency_hint_ms=col["latency_hint_ms"],
            )

    def _make_tool_fn(self, col_config: dict) -> Callable:
        def tool_fn(context: dict[str, Any]) -> dict[str, Any]:
            # Build query from template + context
            params = col_config["extract_query_params"](context)
            query_text = col_config["query_template"].format(**params)

            # Query vector DB
            collection = self.client.get_collection(col_config["collection"])
            results = collection.query(
                query_texts=[query_text],
                n_results=col_config["top_k"],
                where=col_config["metadata_filter"],
            )

            # Transform to standard format
            documents = []
            for i, doc in enumerate(results.get("documents", [[]])[0]):
                meta = results.get("metadatas", [[]])[0][i] if results.get("metadatas") else {}
                score = results.get("distances", [[]])[0][i] if results.get("distances") else 0
                if score >= col_config["score_threshold"]:
                    documents.append({
                        "content": doc,
                        "metadata": meta,
                        "relevance_score": score,
                    })

            return {
                "query": query_text,
                "documents": documents,
                "total_results": len(documents),
                "collection": col_config["collection"],
            }

        return tool_fn


# ═══════════════════════════════════════════════════════════════════════
# 3. MCP PROVIDERS — dynamic tool discovery via Model Context Protocol
# ═══════════════════════════════════════════════════════════════════════

class MCPProvider:
    """
    Wraps one or more MCP servers as dynamically discovered tools.

    Supports three transport modes:
      - stdio:  Server runs as a local subprocess (dev/test)
      - http:   Streamable HTTP transport (production)
      - sse:    Legacy SSE transport (backward compat)

    Discovery flow:
      1. Connect to MCP server
      2. list_tools() → register each as a ToolRegistry callable
      3. list_resources() → register each as a read-only data source
      4. Tool calls route through session.call_tool()
      5. Resource reads route through session.read_resource()

    The provider maintains a persistent session for the lifetime of the
    workflow. Tools are synchronous wrappers around the async MCP SDK
    (using asyncio.run_coroutine_threadsafe or loop.run_until_complete).

    Example:
        # Stdio (dev — runs server as subprocess)
        provider = MCPProvider(
            transport="stdio",
            command="python",
            args=["servers/compliance_server.py"],
            prefix="compliance",
        )

        # Streamable HTTP (production)
        provider = MCPProvider(
            transport="http",
            url="https://compliance-mcp.internal/mcp",
            headers={"Authorization": "Bearer ${COMPLIANCE_TOKEN}"},
            prefix="compliance",
        )

        async with provider:
            provider.register_all(registry)
            # registry now has compliance.search_regulations, etc.
            # ... run workflow ...
    """

    def __init__(
        self,
        transport: str = "stdio",
        # stdio params
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        # http params
        url: str | None = None,
        headers: dict[str, str] | None = None,
        timeout_seconds: float = 30.0,
        # common
        prefix: str = "",
        include_resources: bool = True,
        tool_filter: list[str] | None = None,
    ):
        self.transport = transport
        self.command = command
        self.args = args or []
        self.env = env
        self.url = url
        self.headers = headers or {}
        self.timeout_seconds = timeout_seconds
        self.prefix = prefix
        self.include_resources = include_resources
        self.tool_filter = tool_filter

        # Runtime state (set during connect)
        self._session = None
        self._exit_stack = None
        self._discovered_tools: list[dict] = []
        self._discovered_resources: list[dict] = []
        self._loop = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *exc):
        await self.disconnect()

    async def connect(self):
        """
        Establish connection to MCP server and discover capabilities.

        After connect():
          - self._discovered_tools has all tools with schemas
          - self._discovered_resources has all resources
          - self._session is ready for call_tool / read_resource
        """
        import contextlib
        self._exit_stack = contextlib.AsyncExitStack()
        await self._exit_stack.__aenter__()

        if self.transport == "stdio":
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            server_params = StdioServerParameters(
                command=self.command,
                args=self.args,
                env=self.env,
            )
            read, write = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )

        elif self.transport in ("http", "streamable_http"):
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client

            read, write, _ = await self._exit_stack.enter_async_context(
                streamablehttp_client(
                    self.url,
                    headers=self.headers,
                    timeout=self.timeout_seconds,
                )
            )

        elif self.transport == "sse":
            from mcp import ClientSession
            from mcp.client.sse import sse_client

            read, write = await self._exit_stack.enter_async_context(
                sse_client(self.url, headers=self.headers)
            )

        else:
            raise ValueError(
                f"Unknown transport: {self.transport}. "
                f"Use 'stdio', 'http', or 'sse'."
            )

        from mcp import ClientSession
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()

        # Discover tools
        tools_result = await self._session.list_tools()
        for tool in tools_result.tools:
            if self.tool_filter and tool.name not in self.tool_filter:
                continue
            self._discovered_tools.append({
                "name": tool.name,
                "description": getattr(tool, "description", "") or "",
                "input_schema": getattr(tool, "inputSchema", {}) or {},
            })

        # Discover resources
        if self.include_resources:
            try:
                resources_result = await self._session.list_resources()
                for resource in resources_result.resources:
                    self._discovered_resources.append({
                        "uri": str(resource.uri),
                        "name": getattr(resource, "name", "") or str(resource.uri),
                        "description": getattr(resource, "description", "") or "",
                        "mime_type": getattr(resource, "mimeType", "text/plain"),
                    })
            except Exception:
                # Server may not support resources — that's fine
                pass

    async def disconnect(self):
        """Close the MCP session and transport."""
        if self._exit_stack:
            await self._exit_stack.__aexit__(None, None, None)
            self._exit_stack = None
            self._session = None

    def register_all(self, registry: ToolRegistry):
        """
        Register all discovered tools and resources into a ToolRegistry.

        Tools are registered as callable functions that invoke
        session.call_tool() synchronously (blocking the calling thread
        while the async call completes).

        Resources are registered as read-only tools that return the
        resource content.
        """
        for tool in self._discovered_tools:
            name = f"{self.prefix}.{tool['name']}" if self.prefix else tool["name"]
            fn = self._make_tool_fn(tool)
            # Build description with schema info for the LLM
            desc = tool["description"]
            schema = tool.get("input_schema", {})
            if schema.get("properties"):
                params = ", ".join(schema["properties"].keys())
                desc += f" (params: {params})"
            registry.register(
                name=name,
                fn=fn,
                description=desc,
                latency_hint_ms=500,
            )

        for resource in self._discovered_resources:
            name = f"{self.prefix}.resource.{resource['name']}" if self.prefix else f"resource.{resource['name']}"
            fn = self._make_resource_fn(resource)
            registry.register(
                name=name,
                fn=fn,
                description=f"[Resource] {resource['description']} ({resource['mime_type']})",
                latency_hint_ms=200,
            )

    def _make_tool_fn(self, tool_spec: dict) -> Callable:
        """
        Create a synchronous tool function that calls the MCP server.

        The Retrieve primitive's node runs in a sync context (LangGraph
        nodes are sync). This wrapper bridges to the async MCP session.
        """
        session = self._session
        tool_name = tool_spec["name"]
        input_schema = tool_spec.get("input_schema", {})

        def tool_fn(context: dict[str, Any]) -> dict[str, Any]:
            import asyncio

            # Extract arguments from context based on the tool's input schema
            arguments = {}
            props = input_schema.get("properties", {})
            required = input_schema.get("required", [])
            for param_name in props:
                if param_name in context:
                    arguments[param_name] = context[param_name]
                elif param_name in required:
                    # Try to find it in nested step outputs
                    for key, val in context.items():
                        if isinstance(val, dict) and param_name in val:
                            arguments[param_name] = val[param_name]
                            break

            async def _call():
                result = await session.call_tool(tool_name, arguments=arguments)
                # Extract content from MCP CallToolResult
                contents = []
                for block in result.content:
                    if hasattr(block, "text"):
                        contents.append(block.text)
                    elif hasattr(block, "data"):
                        contents.append(block.data)

                # Try to parse as JSON if it looks like JSON
                combined = "\n".join(str(c) for c in contents)
                try:
                    return json.loads(combined)
                except (json.JSONDecodeError, TypeError):
                    return {"content": combined, "raw_blocks": len(result.content)}

            # Run async in the current or new event loop
            try:
                loop = asyncio.get_running_loop()
                # Already in async context — use create_task
                import concurrent.futures
                future = asyncio.run_coroutine_threadsafe(_call(), loop)
                return future.result(timeout=60)
            except RuntimeError:
                # No running loop — create one
                return asyncio.run(_call())

        return tool_fn

    def _make_resource_fn(self, resource_spec: dict) -> Callable:
        """Create a tool function that reads an MCP resource."""
        session = self._session
        uri = resource_spec["uri"]

        def resource_fn(context: dict[str, Any]) -> dict[str, Any]:
            import asyncio

            async def _read():
                result = await session.read_resource(uri)
                contents = []
                for block in result.contents:
                    if hasattr(block, "text"):
                        contents.append(block.text)
                    elif hasattr(block, "blob"):
                        contents.append(f"[binary: {len(block.blob)} bytes]")
                combined = "\n".join(contents)
                try:
                    return json.loads(combined)
                except (json.JSONDecodeError, TypeError):
                    return {
                        "content": combined,
                        "uri": uri,
                        "mime_type": resource_spec.get("mime_type", "text/plain"),
                    }

            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                future = asyncio.run_coroutine_threadsafe(_read(), loop)
                return future.result(timeout=30)
            except RuntimeError:
                return asyncio.run(_read())

        return resource_fn

    @property
    def tools(self) -> list[dict]:
        """Discovered tools (available after connect)."""
        return self._discovered_tools

    @property
    def resources(self) -> list[dict]:
        """Discovered resources (available after connect)."""
        return self._discovered_resources

    def describe(self) -> str:
        """Human-readable summary of discovered capabilities."""
        lines = [f"MCP Server ({self.transport}): {self.url or self.command}"]
        if self._discovered_tools:
            lines.append(f"  Tools ({len(self._discovered_tools)}):")
            for t in self._discovered_tools:
                lines.append(f"    - {t['name']}: {t['description'][:80]}")
        if self._discovered_resources:
            lines.append(f"  Resources ({len(self._discovered_resources)}):")
            for r in self._discovered_resources:
                lines.append(f"    - {r['name']}: {r['description'][:80]}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# MULTI-SERVER MCP — connect to multiple servers at once
# ═══════════════════════════════════════════════════════════════════════

class MCPMultiProvider:
    """
    Manages connections to multiple MCP servers simultaneously.

    In production, a workflow might need tools from several servers:
      - compliance server (regulation search, policy checks)
      - member services server (member 360, communication log)
      - document server (template retrieval, document generation)

    Example:
        multi = MCPMultiProvider({
            "compliance": {
                "transport": "http",
                "url": "https://compliance-mcp.internal/mcp",
                "prefix": "compliance",
            },
            "member": {
                "transport": "http",
                "url": "https://member-mcp.internal/mcp",
                "prefix": "member",
            },
            "docs": {
                "transport": "stdio",
                "command": "python",
                "args": ["servers/doc_server.py"],
                "prefix": "docs",
            },
        })

        async with multi:
            multi.register_all(registry)
            # registry now has compliance.*, member.*, docs.*
    """

    def __init__(self, servers: dict[str, dict[str, Any]]):
        self._server_configs = servers
        self._providers: dict[str, MCPProvider] = {}

    async def __aenter__(self):
        await self.connect_all()
        return self

    async def __aexit__(self, *exc):
        await self.disconnect_all()

    async def connect_all(self):
        """Connect to all configured MCP servers."""
        for name, config in self._server_configs.items():
            provider = MCPProvider(**config)
            await provider.connect()
            self._providers[name] = provider

    async def disconnect_all(self):
        """Disconnect from all MCP servers."""
        for provider in self._providers.values():
            try:
                await provider.disconnect()
            except Exception:
                pass
        self._providers.clear()

    def register_all(self, registry: ToolRegistry):
        """Register tools from all connected servers."""
        for provider in self._providers.values():
            provider.register_all(registry)

    def describe(self) -> str:
        """Describe all connected servers."""
        return "\n\n".join(p.describe() for p in self._providers.values())

    @property
    def providers(self) -> dict[str, MCPProvider]:
        return self._providers


# ═══════════════════════════════════════════════════════════════════════
# FACTORY — build a production registry from config
# ═══════════════════════════════════════════════════════════════════════

async def build_production_registry(config: dict[str, Any]) -> tuple[ToolRegistry, list]:
    """
    Build a ToolRegistry from a deployment configuration.
    Returns (registry, providers_to_close) — caller must close providers.

    Example config:
    {
        "apis": [
            {
                "base_url": "https://core-banking.internal",
                "auth_env": "CORE_BANKING_TOKEN",
                "endpoints": [
                    {
                        "name": "member_profile",
                        "path": "/v1/members/{member_id}",
                        "description": "Member profile"
                    }
                ]
            }
        ],
        "vectors": [
            {
                "host": "vector-db.internal",
                "collections": [
                    {
                        "name": "prior_disputes",
                        "collection": "dispute_outcomes",
                        "query_template": "{dispute_type} {merchant}",
                        "top_k": 5
                    }
                ]
            }
        ],
        "mcp_servers": {
            "compliance": {
                "transport": "http",
                "url": "https://compliance-mcp.internal/mcp",
                "prefix": "compliance"
            },
            "member_services": {
                "transport": "http",
                "url": "https://member-mcp.internal/mcp",
                "prefix": "member"
            }
        }
    }
    """
    import os
    registry = ToolRegistry()
    providers_to_close = []

    # Wire up API providers
    for api_config in config.get("apis", []):
        token = os.environ.get(api_config.get("auth_env", ""), "")
        provider = APIProvider(
            base_url=api_config["base_url"],
            auth_token=token,
        )
        for ep in api_config.get("endpoints", []):
            provider.add_endpoint(**ep)
        provider.register_all(registry)

    # Wire up vector providers
    for vec_config in config.get("vectors", []):
        provider = VectorProvider(client=None)  # wire real client here
        for col in vec_config.get("collections", []):
            provider.add_collection(**col)
        provider.register_all(registry)

    # Wire up MCP servers
    mcp_configs = config.get("mcp_servers", {})
    if mcp_configs:
        multi = MCPMultiProvider(mcp_configs)
        await multi.connect_all()
        multi.register_all(registry)
        providers_to_close.append(multi)

    return registry, providers_to_close
