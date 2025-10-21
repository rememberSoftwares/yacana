import unittest
import os
import tempfile
from yacana import OllamaAgent
from yacana import OpenAiAgent
from yacana import OllamaModelSettings, OpenAiModelSettings


class BaseAgentTest(unittest.TestCase):
    """Base class for agent tests with common setup and utilities."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        # Create temporary directory for test files
        cls.temp_dir = tempfile.mkdtemp()
        
        # Initialize agents with minimal settings
        cls.ollama_settings = OllamaModelSettings(
            temperature=0.1
        )
        
        cls.openai_settings = OpenAiModelSettings(
            temperature=0.1
        )
        
        # Get which agents to run from environment variables
        cls.run_ollama = os.getenv('TEST_OLLAMA', 'true').lower() == 'true'
        cls.run_openai = os.getenv('TEST_OPENAI', 'true').lower() == 'true'
        cls.run_vllm = os.getenv('TEST_VLLM', 'true').lower() == 'true'
        cls.run_lmstudio = os.getenv('TEST_LMSTUDIO', 'true').lower() == 'true'
        
        # Initialize agents based on which ones are enabled
        if cls.run_ollama:
            cls.ollama_agent = OllamaAgent(
                name="Ollama AI Assistant",
                model_name="llama3.1:8b",
                model_settings=cls.ollama_settings,
                system_prompt="You are a helpful AI assistant",
                endpoint="http://127.0.0.1:11434"
            )
            
            cls.ollama_vision_agent = OllamaAgent(
                name="Ollama Vision AI Assistant",
                model_name="llama3.2-vision:11b-instruct-q4_K_M",
                model_settings=cls.ollama_settings,
                system_prompt="You are a helpful AI assistant",
                endpoint="http://127.0.0.1:11434"
            )
        
        if cls.run_vllm:
            cls.vllm_agent = OpenAiAgent(
                name="VLLM AI Assistant",
                model_name="meta-llama/Llama-3.1-8B-Instruct",
                model_settings=cls.openai_settings,
                system_prompt="You are a helpful AI assistant",
                endpoint="http://127.0.0.1:8000/v1",
                api_token="empty",
                runtime_config={"extra_body": {'guided_decoding_backend': 'outlines'}}
            )
            
        if cls.run_openai:
            openai_api_token = os.getenv('OPENAI_API_TOKEN')
            if not openai_api_token:
                raise unittest.SkipTest("OPENAI_API_TOKEN environment variable not set")
            cls.openai_agent = OpenAiAgent(
                name="OpenAI AI Assistant",
                model_name="gpt-4o-mini",
                model_settings=cls.openai_settings,
                system_prompt="You are a helpful AI assistant",
                api_token=openai_api_token
            )

        if cls.run_lmstudio:
            cls.lmstudio_agent = OpenAiAgent(
                name="LMstudio AI Assistant",
                model_name="google/gemma-2-9b",
                model_settings=cls.openai_settings,
                system_prompt="You are a helpful AI assistant",
                endpoint="http://127.0.0.1:1234/v1",
                api_token="empty"
            )
            

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures after running tests."""
        # Clean up temporary files
        for file in os.listdir(cls.temp_dir):
            os.remove(os.path.join(cls.temp_dir, file))
        os.rmdir(cls.temp_dir)

    def get_test_image_path(self, filename):
        """Get the path to a test image file."""
        return os.path.join("tests", "assets", filename) 