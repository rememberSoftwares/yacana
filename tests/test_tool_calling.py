import unittest
import os
from tests.test_base import BaseAgentTest
from yacana import Task, Tool, ToolError, GenericMessage, GenericAgent, ToolType


def add(number_one: int, number_two: int) -> int:
    """Add two numbers together."""
    if type(number_one) is not int or type(number_two) is not int:
        raise ToolError("Both arguments must be integers.")
    return number_one + number_two


def subtract(number_one: int, number_two: int) -> int:
    """Subtract the second number from the first."""
    if type(number_one) is not int or type(number_two) is not int:
        raise ToolError("Both arguments must be integers.")
    return number_one - number_two


def multiply(number_one: int, number_two: int) -> int:
    """Multiply two numbers together."""
    if type(number_one) is not int or type(number_two) is not int:
        raise ToolError("Both arguments must be integers.")
    return number_one * number_two


def divide(number_one: int, number_two: int) -> int:
    """Divide the first number by the second."""
    if type(number_one) is not int or type(number_two) is not int:
        raise ToolError("Both arguments must be integers.")
    if number_two == 0:
        raise ToolError("Division by zero is not allowed.")
    return number_one // number_two


def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The current real time weather from Weather.com in {city} is sunny with a high of 25°C."


def update_server() -> None:
    """Updates the server."""
    return None


def rollback_server() -> None:
    """Rollbacks the server."""
    return None


class TestToolCalling(BaseAgentTest):
    """Test tool calling capabilities of all agent types."""

    def setUp(self):
        """Set up test fixtures before each test."""
        super().setUp()
        # Create tools
        self.addition = Tool("Addition", "Add two integer numbers and returns the result.", add)
        self.subtraction = Tool("Subtraction", "Subtracts two integer numbers and returns the result.", subtract)
        self.multiplication = Tool("Multiplication", "Multiplies two integer numbers and returns the result.", multiply)
        self.division = Tool("Division", "Divides two integer numbers and returns the result.", divide)
        self.get_weather = Tool("Get_weather", "Calls a weather API and returns the current weather in the given city.", get_weather, shush=True)
        self.update_server = Tool("update_server", "Triggers a server update.", update_server)
        self.rollback_server = Tool("rollback_server", "Triggers a server rollback.", rollback_server)

    def test_basic_arithmetic(self):
        """Test basic arithmetic operations using tools."""

        def test_yacana_tool_calling(agent: GenericAgent):
            agent.history.clean()
            Tool.bulk_tool_type_update([self.addition, self.subtraction, self.multiplication, self.division], ToolType.YACANA)
            prompt = "Calculate 2+4-6*7 by decomposing the operations step by step and according to order of operations (PEMDAS/BODMAS). Use the provided tools. Do not make the math yourself. Only use the tools."
            message = Task(prompt, agent, tools=[self.addition, self.subtraction, self.multiplication, self.division]).solve()
            self.assertTrue(
                any(result in message.content for result in ["-36", "40"]),
                f"Expected either -36 or 40 in the result, got: {message.content}"
            )

        def test_openai_tool_calling(agent: GenericAgent):
            agent.history.clean()
            Tool.bulk_tool_type_update([self.addition, self.subtraction, self.multiplication, self.division], ToolType.OPENAI)
            Task(f"What's `2+4-6*7` ? You can only do one operation at a time. Don't worry you will be ask to continue with the operations later. Follow PEMDAS to solve correctly the equation. You are not allowed to guess the result of any operation. Wait for each tool result before continuing. Only call one tool at a time. When you believe you're finished output 'FINISH'.", agent, tools=[self.addition, self.subtraction, self.multiplication, self.division]).solve()
            while True:
                msg = Task(f"Continue solving the equation! You can only do one operation at a time. Don't worry you will be ask to continue with the operations later. Follow PEMDAS to solve correctly the equation. Only call one tool at a time. When you believe you're finished output 'FINISH'", agent, tools=[self.addition, self.subtraction, self.multiplication, self.division]).solve()
                if "finish" in msg.content.lower():
                    break

            one_before_last_message: GenericMessage = agent.history.get_message(len(agent.history.get_all_messages()) - 2)
            last_message: GenericMessage = agent.history.get_last_message()
            self.assertTrue(
                any(result in one_before_last_message.content for result in ["-36", "40"]) or
                any(result in last_message.content for result in ["-36", "40"]),
                f"Expected either -36 or 40 in the result, got: {one_before_last_message.content} or {last_message.content}"
            )

        # Test Ollama agent
        if self.run_ollama:
            test_yacana_tool_calling(self.ollama_agent)
            test_openai_tool_calling(self.ollama_agent)
        
        # Test OpenAI agent
        if self.run_openai:
            test_yacana_tool_calling(self.openai_agent)
            test_openai_tool_calling(self.openai_agent)
        
        # Test VLLM agent
        if self.run_vllm:
            test_yacana_tool_calling(self.vllm_agent)
            test_openai_tool_calling(self.vllm_agent)

        # Test LMStudio agent
        if self.run_lmstudio:
            test_yacana_tool_calling(self.lmstudio_agent)
            test_openai_tool_calling(self.lmstudio_agent)

    def test_weather_tool(self):
        """Test weather information tool."""
        prompt = "What's the weather in Paris?"
        
        # Test Ollama agent
        if self.run_ollama:
            self.get_weather.tool_type = ToolType.YACANA
            message = Task(prompt, self.ollama_agent, tools=[self.get_weather]).solve()
            self.assertTrue(all([
                "Paris" in message.content,
                "sunny" in message.content.lower(),
                "25" in message.content
            ]))

            self.ollama_agent.history.clean()

            self.get_weather.tool_type = ToolType.OPENAI
            message = Task(prompt, self.ollama_agent, tools=[self.get_weather]).solve()
            self.assertTrue(all([
                "Paris" in message.content,
                "sunny" in message.content.lower(),
                "25" in message.content
            ]))
        
        # Test OpenAI agent
        if self.run_openai:
            message = Task(prompt, self.openai_agent, tools=[self.get_weather]).solve()
            self.assertTrue(all([
                "Paris" in message.content,
                "sunny" in message.content.lower(),
                "25" in message.content
            ]))

            self.openai_agent.history.clean()

            self.get_weather.tool_type = ToolType.OPENAI
            message = Task(prompt, self.openai_agent, tools=[self.get_weather]).solve()
            self.assertTrue(all([
                "Paris" in message.content,
                "sunny" in message.content.lower(),
                "25" in message.content
            ]))
        
        # Test VLLM agent
        if self.run_vllm:
            message = Task(prompt, self.vllm_agent, tools=[self.get_weather]).solve()
            self.assertTrue(all([
                "Paris" in message.content,
                "sunny" in message.content.lower(),
                "25" in message.content
            ]))

            self.vllm_agent.history.clean()

            self.get_weather.tool_type = ToolType.OPENAI
            message = Task(prompt, self.vllm_agent, tools=[self.get_weather]).solve()
            self.assertTrue(all([
                "Paris" in message.content,
                "sunny" in message.content.lower(),
                "25" in message.content
            ]))

        # Test LMSTUDIO agent
        if self.run_lmstudio:
            message = Task(prompt, self.lmstudio_agent, tools=[self.get_weather]).solve()
            self.assertTrue(all([
                "Paris" in message.content,
                "sunny" in message.content.lower(),
                "25" in message.content
            ]))

            self.lmstudio_agent.history.clean()

            self.get_weather.tool_type = ToolType.OPENAI
            message = Task(prompt, self.lmstudio_agent, tools=[self.get_weather]).solve()
            self.assertTrue(all([
                "Paris" in message.content,
                "sunny" in message.content.lower(),
                "25" in message.content
            ]))

    def test_solo_empty_tool(self):
        """Test empty tool to check that no parameter on the called function doesn't break anything. Providing ONE tool only."""
        prompt = "Update the server."

        # Test Ollama agent
        if self.run_ollama:
            self.update_server.tool_type = ToolType.YACANA
            Task(prompt, self.ollama_agent, tools=[self.update_server]).solve()

            self.ollama_agent.history.clean()

            self.update_server.tool_type = ToolType.OPENAI
            Task(prompt, self.ollama_agent, tools=[self.update_server]).solve()

        # Test LMStudio agent
        if self.run_lmstudio:
            self.update_server.tool_type = ToolType.YACANA
            Task(prompt, self.lmstudio_agent, tools=[self.update_server]).solve()

            self.lmstudio_agent.history.clean()

            self.update_server.tool_type = ToolType.OPENAI
            Task(prompt, self.lmstudio_agent, tools=[self.update_server]).solve()


    def test_multi_empty_tools(self):
        """Test empty tool to check that no parameter on the called function doesn't break anything. Providing multiple tools."""
        prompt = "Update the server."

        # Test Ollama agent
        if self.run_ollama:
            self.update_server.tool_type = ToolType.YACANA
            self.rollback_server.tool_type = ToolType.YACANA
            Task(prompt, self.ollama_agent, tools=[self.update_server, self.rollback_server]).solve()

            self.ollama_agent.history.clean()

            self.update_server.tool_type = ToolType.OPENAI
            self.rollback_server.tool_type = ToolType.OPENAI
            Task(prompt, self.ollama_agent, tools=[self.update_server, self.rollback_server]).solve()

        # Test LMStudio agent
        if self.run_lmstudio:
            self.update_server.tool_type = ToolType.YACANA
            self.rollback_server.tool_type = ToolType.YACANA
            Task(prompt, self.lmstudio_agent, tools=[self.update_server, self.rollback_server]).solve()

            self.lmstudio_agent.history.clean()

            self.update_server.tool_type = ToolType.OPENAI
            self.rollback_server.tool_type = ToolType.OPENAI
            Task(prompt, self.lmstudio_agent, tools=[self.update_server, self.rollback_server]).solve()


    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        super().setUpClass()
        
        # Get which agents to run from environment variables
        cls.run_ollama = os.getenv('TEST_OLLAMA', 'true').lower() == 'true'
        cls.run_openai = os.getenv('TEST_OPENAI', 'true').lower() == 'true'
        cls.run_vllm = os.getenv('TEST_VLLM', 'true').lower() == 'true'
        cls.run_vllm = os.getenv('TEST_LMSTUDIO', 'true').lower() == 'true'

if __name__ == '__main__':
    unittest.main()
