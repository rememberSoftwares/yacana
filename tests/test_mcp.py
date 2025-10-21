import unittest
import os
from tests.test_base import BaseAgentTest
from yacana import Task
from yacana.mcp import Mcp
from yacana.tool import ToolType


class TestMcp(BaseAgentTest):
    """Test basic inference capabilities of all agent types."""

    def setUp(self):
        """Clean up agent histories before each test."""
        super().setUp()
        if self.run_ollama:
            self.ollama_agent.history.clean()
        if self.run_openai:
            self.openai_agent.history.clean()
        if self.run_vllm:
            self.vllm_agent.history.clean()
        if self.run_lmstudio:
            self.lmstudio_agent.history.clean()

    def _get_agents(self):
        agents = []
        if self.run_ollama:
            agents.append(("ollama", self.ollama_agent))
        if self.run_openai:
            agents.append(("openai", self.openai_agent))
        if self.run_vllm:
            agents.append(("vllm", self.vllm_agent))
        if self.run_lmstudio:
            agents.append(("lmstudio", self.vllm_agent))
        return agents

    def _reset_agents(self):
        """Reset agent histories after each test."""
        if self.run_ollama:
            self.ollama_agent.history.clean()
        if self.run_openai:
            self.openai_agent.history.clean()
        if self.run_vllm:
            self.vllm_agent.history.clean()
        if self.run_lmstudio:
            self.lmstudio_agent.history.clean()

    def test_connection(self):
        """Test connection to MCP servers and checks that the number of tools matches the expected number for that server."""
        servers = [
            ("deepwiki", "https://mcp.deepwiki.com/mcp", {"read_wiki_structure", "read_wiki_contents", "ask_question"}),
            ("local", "http://localhost:1337/mcp", {"example_tool", "weather"})
        ]
        for name, url, expected_tools in servers:
            mcp = Mcp(url)
            mcp.connect()
            tool_names = {tool.tool_name for tool in mcp.tools}
            self.assertTrue(expected_tools.issubset(tool_names), f"{name} MCP tools mismatch: {tool_names}")
            mcp.disconnect()

    def test_error_handling(self):
        """
        Test for mixing tool execution types YACANA and OPENAI, which should raise an error.
        """
        mcp = Mcp("https://mcp.deepwiki.com/mcp")
        mcp.connect()
        # On mélange les types de tools
        tools = mcp.get_tools_as(ToolType.OPENAI)
        # On force un tool local de type YACANA
        def dummy_tool(x: int): return x
        from yacana.tool import Tool
        local_tool = Tool("dummy", "desc", dummy_tool, tool_type=ToolType.YACANA)
        mixed_tools = tools + [local_tool]
        for _, agent in self._get_agents():
            with self.assertRaises(Exception):
                Task("Test mixing tool types", agent, tools=mixed_tools).solve()
        mcp.disconnect()

    def test_local_tool_override_mcp_tool(self):
        """
        Test that local tools override MCP tools with the same name.
        """
        mcp = Mcp("http://localhost:1337/mcp")
        mcp.connect()
        # MCP fournit "example_tool"
        def example_tool(name: str):
            return {"result": f"local override: {name}"}
        from yacana.tool import Tool
        local_tool = Tool("example_tool", "Override", example_tool, tool_type=ToolType.OPENAI)
        tools = [local_tool] + mcp.get_tools_as(ToolType.OPENAI)
        for _, agent in self._get_agents():
            msg = Task("Call example_tool with name 'yacana'", agent, tools=tools).solve()
            self.assertIn("local override", msg.content)
        mcp.disconnect()

    def test_forget_about_tool(self):
        """
        Test that the forget_tool method works as expected.
        """
        mcp = Mcp("https://mcp.deepwiki.com/mcp")
        mcp.connect()
        self.assertTrue(any(t.tool_name == "read_wiki_structure" for t in mcp.tools))
        mcp.forget_tool("read_wiki_structure")
        self.assertFalse(any(t.tool_name == "read_wiki_structure" for t in mcp.tools))
        mcp.disconnect()

    def test_tool_types(self):
        """
        Test that the tool types are correctly set when getting tools using .get_tools_as() method.
        Also test that mixing tool types in a Task raises IllogicalConfiguration.
        """
        from yacana.tool import IllogicalConfiguration
        mcp = Mcp("http://localhost:1337/mcp")
        mcp.connect()
        # Vérifie que tous les outils sont bien du type demandé
        for tool_type in [ToolType.OPENAI, ToolType.YACANA]:
            tools = mcp.get_tools_as(tool_type)
            for t in tools:
                self.assertEqual(t.tool_type, tool_type)
        # Teste qu'on ne peut pas mélanger les types d'outils dans une Task
        tools_openai = mcp.get_tools_as(ToolType.OPENAI)
        tools_yacana = mcp.get_tools_as(ToolType.YACANA)
        if tools_openai and tools_yacana:
            # Change dynamiquement le type d'un outil pour provoquer l'erreur
            mixed_tools = [tools_openai[0], tools_yacana[0]]
            # Force un type différent sur le deuxième outil
            mixed_tools[1].tool_type = ToolType.YACANA if mixed_tools[0].tool_type == ToolType.OPENAI else ToolType.OPENAI
            for _, agent in self._get_agents():
                with self.assertRaises(IllogicalConfiguration):
                    Task("Test mixing tool types", agent, tools=mixed_tools).solve()
        mcp.disconnect()

    def test_no_tools_are_used(self):
        """If the request doesn't match any given tools, the agent should not use a tool and answer directly."""
        mcp = Mcp("https://mcp.deepwiki.com/mcp")
        mcp.connect()
        tools = mcp.get_tools_as(ToolType.OPENAI)
        for _, agent in self._get_agents():
            msg = Task("What is the capital of France?", agent, tools=tools).solve()
            # On s'attend à une réponse directe, pas à un appel d'outil
            self.assertIn("Paris", msg.content)
        mcp.disconnect()

    def test_tools(self):
        """
        Calls the tools and make sure the output is as expected.
        Use tool execution types YACANA and OPENAI to test the tools.
        Use agent type Ollama and OpenAi to test the tools.
        """
        # On teste sur le serveur local pour éviter les quotas
        mcp = Mcp("http://localhost:1337/mcp")
        mcp.connect()
        # Test avec les deux types d'exécution
        for tool_type in [ToolType.OPENAI, ToolType.YACANA]:
            tools = mcp.get_tools_as(tool_type)
            for _, agent in self._get_agents():
                self._reset_agents()
                # Test de l'outil "weather"
                msg = Task("What is the weather in Paris?", agent, tools=tools).solve()
                self.assertTrue("Paris" in msg.content or "weather" in msg.content.lower())
                # Test de l'outil "example_tool"
                msg2 = Task("Call example_tool with name 'yacana'", agent, tools=tools).solve()
                self.assertIn("yacana", msg2.content.lower())
        mcp.disconnect()

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        super().setUpClass()
        # Get which agents to run from environment variables
        cls.run_ollama = os.getenv('TEST_OLLAMA', 'true').lower() == 'true'
        cls.run_openai = os.getenv('TEST_OPENAI', 'true').lower() == 'true'
        cls.run_vllm = os.getenv('TEST_VLLM', 'true').lower() == 'true'
        cls.run_lmstudio = os.getenv('TEST_LMSTUDIO', 'true').lower() == 'true'

if __name__ == '__main__':
    unittest.main()
