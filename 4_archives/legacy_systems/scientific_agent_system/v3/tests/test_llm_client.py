import pytest
import os
import json
import urllib.error
from unittest.mock import patch, MagicMock
from llm_client import LLMClient

# Fixture
@pytest.fixture
def client():
    return LLMClient()

def test_init_defaults(client):
    assert client.model == "gpt-4o"
    # Depends on environment, but defaults should handle None safely

def test_chat_completion_no_key_fallback(client):
    # Ensure no API key
    client.api_key = None
    messages = [{"role": "user", "content": "Hello"}]
    
    # Needs to match specific Turn 1/2 logic in mock_fallback or default
    # Since mocked fallback depends on len(messages), let's test that
    response = client.chat_completion(messages)
    assert "THOUGHT:" in response

def test_chat_completion_success(client):
    client.api_key = "test_key"
    mock_gpt_response = {
        "choices": [
            {
                "message": {"content": "Hello there!"}
            }
        ]
    }
    
    with patch('urllib.request.urlopen') as mock_urlopen:
        mock_response_obj = MagicMock()
        mock_response_obj.read.return_value = json.dumps(mock_gpt_response).encode('utf-8')
        mock_response_obj.__enter__.return_value = mock_response_obj
        mock_urlopen.return_value = mock_response_obj
        
        response = client.chat_completion([{"role": "user", "content": "Hi"}])
        assert response == "Hello there!"

def test_chat_completion_http_error(client):
    client.api_key = "test_key"
    
    with patch('urllib.request.urlopen') as mock_urlopen:
        # Simulate HTTPError
        err = urllib.error.HTTPError(
            url="http://test", 
            code=401, 
            msg="Unauthorized", 
            hdrs={}, 
            fp=MagicMock()
        )
        err.read = MagicMock(return_value=b"Invalid API Key")
        mock_urlopen.side_effect = err
        
        # Need to import urllib.error in test execution context or patch where it's used
        # Since we are mocking the side effect, the exception will be raised.
        # But wait, LLMClient catches it.
        
        response = client.chat_completion([{"role": "user", "content": "Hi"}])
        assert "Error calling LLM: 401" in response
        assert "Invalid API Key" in response

def test_mock_fallback_logic(client):
    # Turn 1
    msgs_1 = [{"role": "user", "content": "start"}] * 2 # len 2
    resp_1 = client._mock_fallback(msgs_1)
    assert "create_file" in resp_1
    
    # Turn 2
    msgs_2 = [{"role": "user", "content": "next"}] * 4 # len 4
    resp_2 = client._mock_fallback(msgs_2)
    assert "search_web" in resp_2
    
    # Turn 3
    msgs_3 = [{"role": "user", "content": "end"}] * 6 # len 6
    resp_3 = client._mock_fallback(msgs_3)
    assert "Stop" in resp_3
