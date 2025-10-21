import unittest
import os
from tests.test_base import BaseAgentTest
from yacana import Task, Message, MessageRole, History, HistorySlot, OpenAiModelSettings, Tool
from yacana.generic_agent import GenericAgent

class TestBasicInference(BaseAgentTest):
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

    def test_simple_completion(self):
        """Test basic text completion with all agent types."""
        prompt = "Count from 1 to 5 (no additionnal text, numbers only):"
        expected = "1, 2, 3, 4, 5"
        
        # Test Ollama agent
        if self.run_ollama:
            message = Task(prompt, self.ollama_agent).solve()
            self.assertIn("1", message.content)
            self.assertIn("5", message.content)
        
        # Test OpenAI agent
        if self.run_openai:
            message = Task(prompt, self.openai_agent).solve()
            self.assertIn("1", message.content)
            self.assertIn("5", message.content)
        
        # Test VLLM agent
        if self.run_vllm:
            message = Task(prompt, self.vllm_agent).solve()
            self.assertIn("1", message.content)
            self.assertIn("5", message.content)

    def test_history_management(self):
        """Test basic history management operations."""
        history = History()
        
        # Test adding messages using Message class
        user_message = Message(MessageRole.USER, "Hello, how are you?")
        history.add_message(user_message)
        
        # Test adding messages using Task with all agents
        if self.run_ollama:
            assistant_message = Task("Respond to the greeting", self.ollama_agent).solve()
            history.add_message(assistant_message)
        
        if self.run_openai:
            assistant_message = Task("Respond to the greeting", self.openai_agent).solve()
            history.add_message(assistant_message)
        
        if self.run_vllm:
            assistant_message = Task("Respond to the greeting", self.vllm_agent).solve()
            history.add_message(assistant_message)
        
        # Verify history structure
        expected_slots = sum([self.run_ollama, self.run_openai, self.run_vllm]) + 1  # +1 for user message
        self.assertEqual(len(history.slots), expected_slots)
        
        # Test getting messages as dictionary
        messages_dict = history.get_messages_as_dict()
        self.assertEqual(len(messages_dict), expected_slots)
        self.assertEqual(messages_dict[0]["role"], "user")
        for i in range(1, expected_slots):
            self.assertEqual(messages_dict[i]["role"], "assistant")

    def test_slot_management(self):
        """Test slot management operations."""
        history = History()
        
        # Create a slot with multiple messages
        slot = HistorySlot()
        user_message = Message(MessageRole.USER, "What's the weather like?")
        slot.add_message(user_message)
        
        # Add slot to history
        history.add_slot(slot)
        
        # Test changing default message with all agents
        if self.run_ollama:
            new_message = Task("Describe the weather", self.ollama_agent).solve()
            slot.add_message(new_message)
        
        if self.run_openai:
            new_message = Task("Describe the weather", self.openai_agent).solve()
            slot.add_message(new_message)
        
        if self.run_vllm:
            new_message = Task("Describe the weather", self.vllm_agent).solve()
            slot.add_message(new_message)
        
        # Change the selected message and verify
        expected_messages = sum([self.run_ollama, self.run_openai, self.run_vllm]) + 1  # +1 for user message
        self.assertEqual(len(slot.messages), expected_messages)
        
        # Test keeping only selected message
        slot.set_main_message_index(1)  # Select the first assistant message
        slot.keep_only_selected_message()
        self.assertEqual(len(slot.messages), 1)
        self.assertEqual(slot.get_message().role, MessageRole.ASSISTANT)

    def test_message_labelling(self):
        """Test message labelling system."""
        history = History()
        
        # Create messages with tags for all agents
        user_message = Message(MessageRole.USER, "What's the weather like?", tags=["weather", "query"])
        history.add_message(user_message)

        if self.run_ollama:
            assistant_message = Task("Describe the weather", self.ollama_agent).solve()
            assistant_message.add_tags(["weather"])
            history.add_message(assistant_message)
        
        if self.run_openai:
            assistant_message = Task("Describe the weather", self.openai_agent).solve()
            assistant_message.add_tags(["weather"])
            history.add_message(assistant_message)
        
        if self.run_vllm:
            assistant_message = Task("Describe the weather", self.vllm_agent).solve()
            assistant_message.add_tags(["weather"])
            history.add_message(assistant_message)
        
        # Test getting messages by tags
        weather_messages = history.get_messages_by_tags(["weather"])
        expected_messages = sum([self.run_ollama, self.run_openai, self.run_vllm]) + 1  # +1 for user message
        self.assertEqual(len(weather_messages), expected_messages)
        
        # Test strict tag matching
        weather_messages = history.get_messages_by_tags(["weather", "nonexistent"], strict=True)
        self.assertEqual(len(weather_messages), 0)
        
        # Test non-strict tag matching
        weather_messages = history.get_messages_by_tags(["weather", "nonexistent"], strict=False)
        self.assertEqual(len(weather_messages), expected_messages)

    def test_tag_matching_modes(self):
        """Test the different tag matching modes of get_messages_by_tags."""
        history = History()
        
        # Create messages with different tag combinations
        message1 = Message(MessageRole.USER, "Message 1", tags=["tag1"])
        message2 = Message(MessageRole.USER, "Message 2", tags=["tag1", "tag2"])
        message3 = Message(MessageRole.USER, "Message 3", tags=["tag2", "tag3"])
        message4 = Message(MessageRole.USER, "Message 4", tags=["tag1", "tag2", "tag3"])
        
        # Add messages to history
        history.add_message(message1)
        history.add_message(message2)
        history.add_message(message3)
        history.add_message(message4)
        
        # Test non-strict mode (default) - should match ANY of the specified tags
        # Test with single tag
        messages = history.get_messages_by_tags(["tag1"])
        self.assertEqual(len(messages), 3)  # Should match message1, message2, message4
        self.assertIn(message1, messages)
        self.assertIn(message2, messages)
        self.assertIn(message4, messages)
        self.assertNotIn(message3, messages)
        
        # Test with multiple tags in non-strict mode
        messages = history.get_messages_by_tags(["tag1", "tag3"])
        self.assertEqual(len(messages), 4)  # Should match all messages except those with no matching tags
        self.assertIn(message1, messages)
        self.assertIn(message2, messages)
        self.assertIn(message3, messages)
        self.assertIn(message4, messages)
        
        # Test strict mode - should match messages that have ALL specified tags
        # Test with single tag (should behave same as non-strict)
        messages = history.get_messages_by_tags(["tag1"], strict=True)
        self.assertEqual(len(messages), 3)  # Should match message1, message2, message4
        self.assertIn(message1, messages)
        self.assertIn(message2, messages)
        self.assertIn(message4, messages)
        self.assertNotIn(message3, messages)
        
        # Test with multiple tags in strict mode
        messages = history.get_messages_by_tags(["tag1", "tag2"], strict=True)
        self.assertEqual(len(messages), 2)  # Should match only message2 and message4
        self.assertIn(message2, messages)
        self.assertIn(message4, messages)
        self.assertNotIn(message1, messages)
        self.assertNotIn(message3, messages)
        
        # Test with non-existent tag
        messages = history.get_messages_by_tags(["nonexistent"], strict=True)
        self.assertEqual(len(messages), 0)
        
        # Test with combination of existing and non-existent tags
        messages = history.get_messages_by_tags(["tag1", "nonexistent"], strict=True)
        self.assertEqual(len(messages), 0)
        
        # Test with empty tag list
        messages = history.get_messages_by_tags([], strict=True)
        self.assertEqual(len(messages), 4)  # Should match all messages when no tags specified

    def test_builtin_tags(self):
        """Test the builtin tag system (yacana_builtin, yacana_prompt, yacana_response)."""
        def test_agent_builtin_tags(agent: GenericAgent):
            # Create a task with a custom tag
            Task("Count from 1 to 3", agent, tags=["custom_tag"]).solve()
            
            # Get the history
            history: History = agent.history
            
            # Test getting prompt messages
            prompt_messages = history.get_messages_by_tags(["yacana_prompt"])
            self.assertEqual(len(prompt_messages), 1, "Should find exactly one prompt message")
            self.assertEqual(prompt_messages[0].role, MessageRole.USER, "Prompt message should be from user")
            
            # Test getting response messages
            response_messages = history.get_messages_by_tags(["yacana_response"])
            self.assertEqual(len(response_messages), 1, "Should find exactly one response message")
            self.assertEqual(response_messages[0].role, MessageRole.ASSISTANT, "Response message should be from assistant")
            
            # Test getting both messages using non-strict matching with both tags (False by default)
            all_messages = history.get_messages_by_tags(["yacana_prompt", "yacana_response"], strict=False)
            self.assertEqual(len(all_messages), 2, "Should find both prompt and response messages")
            
            # Verify the custom tag is also present
            custom_tag_messages = history.get_messages_by_tags(["custom_tag"])
            self.assertEqual(len(custom_tag_messages), 2, "Should find the messages with custom tag")
            self.assertEqual(custom_tag_messages[0].role, MessageRole.USER, "Custom tag should be on user message")
            self.assertEqual(custom_tag_messages[1].role, MessageRole.ASSISTANT, "Custom tag should be on assistant message")

        # Test OpenAI agent
        if self.run_openai:
            test_agent_builtin_tags(self.openai_agent)
        
        # Test VLLM agent
        if self.run_vllm:
            test_agent_builtin_tags(self.vllm_agent)
        
        # Test Ollama agent
        if self.run_ollama:
            test_agent_builtin_tags(self.ollama_agent)

    def test_multiple_responses(self):
        """Test getting multiple responses from the model."""
        # Test OpenAI agent
        if self.run_openai:
            settings = OpenAiModelSettings(temperature=0.1, n=2)
            self.openai_agent.model_settings = settings
            task = Task("Count from 1 to 3", self.openai_agent)
            task.solve()  # We don't need the returned message since we'll check the slot
            
            # Check that we got the correct number of messages in the history slot
            slot = self.openai_agent.history.get_last_slot()
            self.assertEqual(len(slot.messages), 2, "Expected 2 messages in the slot")
            
            # Validate each message's content
            for msg in slot.messages:
                self.assertTrue(
                    any(str(i) in msg.content for i in range(1, 4)),
                    f"Expected numbers 1-3 in response, got: {msg.content}"
                )
        
        # Test VLLM agent (which also supports multiple responses)
        if self.run_vllm:
            settings = OpenAiModelSettings(temperature=0.1, n=2)
            self.vllm_agent.model_settings = settings
            task = Task("Count from 1 to 3", self.vllm_agent)
            task.solve()  # We don't need the returned message since we'll check the slot
            
            # Check that we got the correct number of messages in the history slot
            slot = self.vllm_agent.history.get_last_slot()
            self.assertEqual(len(slot.messages), 2, "Expected 2 messages in the slot")
            
            # Validate each message's content
            for msg in slot.messages:
                self.assertTrue(
                    any(str(i) in msg.content for i in range(1, 4)),
                    f"Expected numbers 1-3 in response, got: {msg.content}"
                )
        
        # Note: Ollama doesn't support multiple responses (n parameter), so we skip it

    def test_forget_history(self):
        """Test that the forget=True option restores history to its initial state."""
        def test_agent_forget(agent: GenericAgent):
            # Add some initial messages to the history
            initial_messages = [
                Message(MessageRole.USER, "Hello"),
                Message(MessageRole.ASSISTANT, "Hi there!")
            ]
            for msg in initial_messages:
                agent.history.add_message(msg)
            
            # Store the initial history state
            initial_history_length = len(agent.history.slots)
            
            # Create and solve a task with forget=True
            task = Task("Count from 1 to 3", agent, forget=True)
            task.solve()
            
            # Verify that the history was restored to its initial state
            self.assertEqual(
                len(agent.history.slots),
                initial_history_length,
                f"{agent.name} history length should be restored to initial state"
            )
            
            # Verify the content of the history matches the initial state
            # Skip the first slot (system prompt) and check the rest
            for i, slot in enumerate(agent.history.slots[1:], start=1):
                self.assertEqual(
                    slot.get_message().content,
                    initial_messages[i-1].content,
                    f"{agent.name} message {i} content should match initial state"
                )
                self.assertEqual(
                    slot.get_message().role,
                    initial_messages[i-1].role,
                    f"{agent.name} message {i} role should match initial state"
                )
        
        # Test OpenAI agent
        if self.run_openai:
            test_agent_forget(self.openai_agent)
        
        # Test VLLM agent
        if self.run_vllm:
            test_agent_forget(self.vllm_agent)
        
        # Test Ollama agent
        if self.run_ollama:
            test_agent_forget(self.ollama_agent)

    def test_history_deletion_methods(self):
        """Test the ability to delete messages and slots from the history."""
        history = History()

        # Ajouter 3 messages dans 3 slots différents
        msg1 = Message(MessageRole.USER, "Message 1")
        msg2 = Message(MessageRole.ASSISTANT, "Message 2")
        msg3 = Message(MessageRole.USER, "Message 3")
        slot1 = history.add_message(msg1)
        slot2 = history.add_message(msg2)
        slot3 = history.add_message(msg3)

        # Vérifier que les 3 slots sont présents
        self.assertEqual(len(history.slots), 3)

        # Supprimer le message 2 via delete_message
        history.delete_message(msg2)
        self.assertEqual(len(history.slots), 2)
        self.assertNotIn(slot2, history.slots)
        self.assertNotIn(msg2, [slot.get_message() for slot in history.slots])

        # Supprimer le message 1 via delete_message_by_id
        history.delete_message_by_id(msg1.id)
        self.assertEqual(len(history.slots), 1)
        self.assertNotIn(slot1, history.slots)
        self.assertNotIn(msg1, [slot.get_message() for slot in history.slots])

        # Ajouter un slot supplémentaire pour tester delete_slot et delete_slot_by_id
        msg4 = Message(MessageRole.ASSISTANT, "Message 4")
        slot4 = history.add_message(msg4)
        self.assertEqual(len(history.slots), 2)

        # Supprimer le slot via delete_slot
        history.delete_slot(slot3)
        self.assertEqual(len(history.slots), 1)
        self.assertNotIn(slot3, history.slots)

        # Supprimer le slot restant via delete_slot_by_id
        history.delete_slot_by_id(slot4.id)
        self.assertEqual(len(history.slots), 0)

    def test_no_structure_thinking(self):
        """Test the ability to not use JSON structured output internally."""

        def update_server() -> None:
            """Updates the server."""
            return None

        def rollback_server() -> None:
            """Rollbacks the server."""
            return None
        update_server = Tool("update_server", "Triggers a server update.", update_server)
        rollback_server = Tool("rollback_server", "Triggers a server rollback.", rollback_server)

        def test(agent: GenericAgent):
            agent.structured_thinking = False
            Task("Update the server", agent, tools=[update_server]).solve()
            Task("Update the server", agent, tools=[update_server, rollback_server]).solve()
            agent.structured_thinking = True
            # Vérifier que le mot "JSON" n'apparaît qu'une seule fois dans les contenus des messages
            all_messages = agent.history.get_all_messages()
            json_count = sum("JSON" in message.content for message in all_messages)
            self.assertEqual(json_count, 2, 'Le mot "JSON" doit apparaître exactement 2 fois dans les contenus des messages (1 par Task)')

        if self.run_openai:
            test(self.openai_agent)
        if self.run_vllm:
            test(self.vllm_agent)
        if self.run_ollama:
            test(self.ollama_agent)
        if self.run_lmstudio:
            test(self.lmstudio_agent)


    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        super().setUpClass()
        
        # Get which agents to run from environment variables
        cls.run_ollama = os.getenv('TEST_OLLAMA', 'true').lower() == 'true'
        cls.run_openai = os.getenv('TEST_OPENAI', 'true').lower() == 'true'
        cls.run_vllm = os.getenv('TEST_VLLM', 'true').lower() == 'true'

if __name__ == '__main__':
    unittest.main() 