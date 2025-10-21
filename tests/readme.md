Start with:
TEST_OLLAMA=true TEST_OPENAI=false TEST_VLLM=false TEST_LMSTUDIO=false python3 -m unittest tests/test_state_persistence.py -v
TEST_OLLAMA=true TEST_OPENAI=false TEST_VLLM=false TEST_LMSTUDIO=false python3 -m unittest tests.test_state_persistence.TestStatePersistence.test_state_with_structured_output -v
export OPENAI_API_TOKEN=sk-proj-XXXXXXXXXXXX
