"""Fakes for the MCP SDK calls inside Core's real ``MCPClient.connect``."""

from contextlib import asynccontextmanager

from open_webui.utils.mcp import client as core_mcp


def patch_mcp_sdk(monkeypatch, connection):
    """Make ``MCPClient.connect(url)`` hold ``connection(url)`` open as its SDK session.

    ``connection(url)`` yields the server instructions.
    """

    class Session:
        instructions = None

        def __init__(self, url):
            self.context = connection(url)

        async def __aenter__(self):
            self.instructions = await self.context.__aenter__()
            return self

        async def __aexit__(self, *exc_info):
            return await self.context.__aexit__(*exc_info)

        async def initialize(self):
            return self

    @asynccontextmanager
    async def http_client(**kwargs):
        yield None

    @asynccontextmanager
    async def streams(url, **kwargs):
        yield url, None, None

    if hasattr(core_mcp, "Client"):
        monkeypatch.setattr(core_mcp, "create_httpx_client", http_client)
        monkeypatch.setattr(core_mcp, "streamable_http_client", lambda url, **kwargs: url)
        monkeypatch.setattr(core_mcp, "Client", lambda url, **kwargs: Session(url))
    else:
        monkeypatch.setattr(core_mcp, "streamablehttp_client", streams)
        monkeypatch.setattr(core_mcp, "ClientSession", lambda url, write, **kwargs: Session(url))
