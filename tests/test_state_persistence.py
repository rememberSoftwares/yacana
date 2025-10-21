import unittest
import os
from tests.test_base import BaseAgentTest
from yacana import Task, Agent
from pydantic import BaseModel


class CountryFact(BaseModel):
    """A fact about a country."""
    name: str
    fact: str


class Facts(BaseModel):
    """Collection of country facts."""
    countryFacts: list[CountryFact]


class TestStatePersistence(BaseAgentTest):
    """Test agent state persistence and restoration."""

    def test_export_import_state(self):
        """Test exporting and importing agent state."""
        def test_state_persistence(agent):
            # Create initial state with some history
            message1 = Task("Tell me 1 fact about Canada.", agent).solve()
            message2 = Task("Tell me 2 facts about Canada.", agent, structured_output=Facts).solve()
            message_id = message2.id
            message2.add_tags(["export_import_version"])
            
            # Export state
            state_file = os.path.join(self.temp_dir, f"{agent.name}_state.json")
            agent.export_to_file(state_file)
            
            # Create new agent and import state
            new_agent = Agent.import_from_file(state_file)
            
            # Test if history is preserved
            messages = new_agent.history.get_messages_by_tags(["export_import_version"])
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0].id, message_id)
            
            # Test if structured output is preserved
            facts = messages[0].structured_output
            self.assertIsInstance(facts, Facts)
            self.assertEqual(len(facts.countryFacts), 2)
            
            # Test if we can continue the conversation
            message3 = Task("Tell me 1 fact about England.", new_agent).solve()
            self.assertGreater(len(message3.content), 0)
        
        # Test Ollama agent
        if self.run_ollama:
            test_state_persistence(self.ollama_agent)
        
        # Test OpenAI agent
        if self.run_openai:
            test_state_persistence(self.openai_agent)
        
        # Test VLLM agent
        if self.run_vllm:
            test_state_persistence(self.vllm_agent)

        # Test LMSTUDIO agent
        if self.run_lmstudio:
            test_state_persistence(self.lmstudio_agent)

    def test_state_with_media(self):
        """Test state persistence with media history."""
        def test_media_state(agent):
            # Create state with media interaction
            image_path = self.get_test_image_path("burger.jpg")
            message1 = Task("Describe this image:", agent, medias=[image_path]).solve()
            message1.add_tags(["media_version"])
            
            # Export state
            state_file = os.path.join(self.temp_dir, f"{agent.name}_media_state.json")
            agent.export_to_file(state_file)
            
            # Import state into new agent
            new_agent = Agent.import_from_file(state_file)
            
            # Test if media history is preserved
            messages = new_agent.history.get_messages_by_tags(["media_version"])
            self.assertEqual(len(messages), 1)
            
            # Test if we can continue the conversation
            message2 = Task("Repeat what you said about the image.", new_agent).solve()
            self.assertTrue(
                any(keyword in message2.content.lower() for keyword in ['food', 'burger', 'meal', 'dish', 'eating']),
                "Response should mention food-related content"
            )
        
        # Test Ollama agent
        if self.run_ollama:
            test_media_state(self.ollama_vision_agent)
        # Test OpenAI agent
        if self.run_openai:
            test_media_state(self.openai_agent)
        # Test VLLM agent
        #if self.run_vllm:
        #    test_media_state(self.vllm_agent)

    def test_state_with_structured_output(self):
        """Test state persistence with structured output history."""
        def test_structured_state(agent):
            # Create state with structured output
            message1 = Task("Tell me 2 facts about Canada.", agent, structured_output=Facts).solve()
            message1.add_tags(["structured_output_version"])
            
            # Export state
            state_file = os.path.join(self.temp_dir, f"{agent.name}_structured_state.json")
            agent.export_to_file(state_file)
            
            # Import state into new agent
            new_agent = Agent.import_from_file(state_file)
            
            # Test if structured output is preserved
            messages = new_agent.history.get_messages_by_tags(["structured_output_version"])
            self.assertEqual(len(messages), 1)
            facts = messages[0].structured_output
            self.assertIsInstance(facts, Facts)
            self.assertEqual(len(facts.countryFacts), 2)
            
            # Test if we can continue with structured output
            message2 = Task("Tell me 2 facts about France.", new_agent, structured_output=Facts).solve()
            self.assertIsInstance(message2.structured_output, Facts)
            self.assertEqual(len(message2.structured_output.countryFacts), 2)  # VLLM has trouble outputting 2 facts which may fail the test
        
        # Test Ollama agent
        if self.run_ollama:
            test_structured_state(self.ollama_agent)
        
        # Test OpenAI agent
        if self.run_openai:
            test_structured_state(self.openai_agent)
        
        # Test VLLM agent
        if self.run_vllm:
            test_structured_state(self.vllm_agent)

        # Test LMSTUDIO agent
        if self.run_lmstudio:
            test_structured_state(self.lmstudio_agent)

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