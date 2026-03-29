"""
Cognitive Core — Fraud Data MCP Server

stdio MCP server that serves fraud case tool data from SQLite.

On startup: seeds the DB from *_tools.json fixture files.
list_tools: returns one MCP Tool per distinct tool_name in DB,
            each requiring a case_id parameter.
call_tool:  looks up (case_id, tool_name) → returns JSON TextContent.

Usage:
    python demos/fraud-operations/fraud_data_mcp.py \\
        --db demos/fraud-operations/fraud_data.db \\
        --fixtures demos/fraud-operations/cases/fixtures
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add project root to path so coordinator/engine imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

from fraud_db import (
    import_fixtures_dir,
    get_tool_data,
    list_tool_names,
)


def build_server(db_path: str) -> Server:
    server = Server("fraud-data")

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        names = list_tool_names(db_path)
        tools = []
        for name in names:
            tools.append(
                types.Tool(
                    name=name,
                    description=f"Retrieve {name} data for a fraud case",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "case_id": {
                                "type": "string",
                                "description": "The fraud case identifier (e.g. FRD-2026-00902)",
                            }
                        },
                        "required": ["case_id"],
                    },
                )
            )
        return tools

    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict
    ) -> list[types.TextContent]:
        case_id = arguments.get("case_id", "")
        if not case_id:
            return [types.TextContent(type="text", text=json.dumps({"error": "case_id is required"}))]

        data = get_tool_data(db_path, case_id, name)
        if data is None:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": f"No data for tool={name} case_id={case_id}"}),
            )]

        return [types.TextContent(type="text", text=json.dumps(data))]

    return server


async def main(db_path: str, fixtures_dir: str | None) -> None:
    # Seed DB from fixtures on startup
    if fixtures_dir:
        inserted = import_fixtures_dir(db_path, fixtures_dir)
        tool_count = len(list_tool_names(db_path))
        print(
            f"fraud-data-mcp: seeded DB from {fixtures_dir} "
            f"({inserted} rows inserted, {tool_count} distinct tools available)",
            file=sys.stderr,
        )

    server = build_server(db_path)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Data MCP Server")
    parser.add_argument("--db", required=True, help="Path to SQLite DB file")
    parser.add_argument("--fixtures", default=None, help="Directory of *_tools.json fixture files to seed on startup")
    args = parser.parse_args()

    asyncio.run(main(args.db, args.fixtures))
