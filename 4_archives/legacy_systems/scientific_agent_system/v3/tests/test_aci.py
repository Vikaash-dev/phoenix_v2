import pytest
import subprocess
import os
import json
from unittest.mock import patch, MagicMock, mock_open
from aci import ACI

# Fixture for ACI instance
@pytest.fixture
def aci():
    return ACI()

def test_init(aci):
    assert aci.is_local is True
    assert aci.runtime is None

def test_run_shell_local_success(aci):
    with patch('subprocess.check_output') as mock_sub:
        mock_sub.return_value = b"Hello World"
        result = aci.run_shell("echo Hello World")
        assert "[SUCCESS]" in result
        assert "Hello World" in result
        mock_sub.assert_called_once_with("echo Hello World", shell=True, stderr=subprocess.STDOUT)

def test_run_shell_local_failure(aci):
    with patch('subprocess.check_output') as mock_sub:
        error = subprocess.CalledProcessError(1, "cmd", output=b"Error output")
        mock_sub.side_effect = error
        result = aci.run_shell("invalid_command")
        assert "[EXIT STATUS 1]" in result
        assert "Error output" in result

def test_create_file_local_success(aci):
    with patch("builtins.open", mock_open()) as mock_file:
        result = aci.create_file("test.txt", "content")
        assert "File test.txt created." in result
        mock_file.assert_called_with("test.txt", "w")
        mock_file().write.assert_called_with("content")

def test_create_file_local_failure(aci):
    with patch("builtins.open", side_effect=IOError("Permission denied")):
        result = aci.create_file("test.txt", "content")
        assert "Failed to create" in result
        assert "Permission denied" in result

def test_read_file_local_success(aci):
    with patch("builtins.open", mock_open(read_data="file content")) as mock_file:
        content = aci.read_file("test.txt")
        assert content == "file content"
        mock_file.assert_called_with("test.txt", "r")

def test_read_file_local_failure(aci):
    with patch("builtins.open", side_effect=FileNotFoundError("No such file")):
        result = aci.read_file("test.txt")
        assert "Error reading file" in result
        assert "No such file" in result

def test_search_web_no_key(aci):
    # Ensure TAVILY_API_KEY is not set
    with patch.dict(os.environ, {}, clear=True):
        result = aci.search_web("test")
        assert "ERROR: TAVILY_API_KEY not found" in result

def test_search_web_success(aci):
    mock_response = {
        "results": [
            {"title": "Test Result", "url": "http://test.com", "content": "This is a test snippet."}
        ]
    }
    
    with patch.dict(os.environ, {"TAVILY_API_KEY": "test_key"}):
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_response_obj = MagicMock()
            mock_response_obj.read.return_value = json.dumps(mock_response).encode('utf-8')
            mock_response_obj.__enter__.return_value = mock_response_obj
            mock_urlopen.return_value = mock_response_obj
            
            result = aci.search_web("query")
            
            assert "Title: Test Result" in result
            assert "URL: http://test.com" in result
            assert "Snippet: This is a test snippet" in result

def test_negative_search(aci):
    with patch.object(aci, 'search_web', return_value="Mock Search Result") as mock_search:
        result = aci.negative_search("theory")
        assert "Performing NEGATIVE ANALYSIS on: 'theory'" in result
        assert "Mock Search Result" in result
        mock_search.assert_called_once()
        args, _ = mock_search.call_args
        assert "limitations" in args[0]
        assert "failures" in args[0]
