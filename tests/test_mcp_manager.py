"""Tests for MCPManager."""

from dual_agent.mcp_manager import MCPManager
from dual_agent.mcp_host import MCPHost


def test_mcp_manager_add_and_remove(tmp_path):
    config_file = str(tmp_path / "mcp_servers.json")
    mgr = MCPManager(config_path=config_file)

    # Initially empty
    servers = mgr.load_servers()
    assert len(servers) == 0

    # Add a server
    mgr.add_server(
        name="github",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={"TOKEN": "xyz"},
    )

    servers = mgr.load_servers()
    assert "github" in servers
    assert servers["github"].command == "npx"
    assert servers["github"].args == ["-y", "@modelcontextprotocol/server-github"]

    # Attach to host
    host = MCPHost()
    count = mgr.attach_to_host(host)
    assert count == 1
    assert host.get_tool("mcp_github_dispatch") is not None

    # Remove server
    removed = mgr.remove_server("github")
    assert removed is True
    assert "github" not in mgr.load_servers()


def test_external_mcp_server_fails_loudly_when_called(tmp_path):
    """Calling an external MCP server placeholder must fail loudly, not claim success."""
    config_file = str(tmp_path / "mcp_servers.json")
    mgr = MCPManager(config_path=config_file)
    mgr.add_server(
        name="test_ext",
        command="external-cmd",
        args=["arg1"],
    )
    host = MCPHost()
    mgr.attach_to_host(host)

    res = host.execute_tool("mcp_test_ext_dispatch", {"action": "list_repos"})
    assert res.success is False
    assert "not connected" in str(res.error).lower()
    assert "not implemented" in str(res.error).lower()
