import unittest
import sys
import os
import re

# Add the parent directory to sys.path to allow importing agent_loop
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent_loop import parse_action

class TestParsing(unittest.TestCase):
    def test_run_shell_simple(self):
        output = "I will run a command.\nACTION: run_shell('ls -la')"
        action, args = parse_action(output)
        self.assertEqual(action, "run_shell")
        self.assertEqual(args, "ls -la")

    def test_run_shell_with_escaped_quotes(self):
        output = r"ACTION: run_shell('echo \'hello\'')"
        action, args = parse_action(output)
        self.assertEqual(action, "run_shell")
        self.assertEqual(args, "echo 'hello'")

    def test_create_file_simple(self):
        output = "ACTION: create_file('test.txt', 'content')"
        action, args = parse_action(output)
        self.assertEqual(action, "create_file")
        self.assertEqual(args, ("test.txt", "content"))

    def test_create_file_multiline(self):
        output = "ACTION: create_file('test.txt', 'line1\nline2')"
        action, args = parse_action(output)
        self.assertEqual(action, "create_file")
        self.assertEqual(args, ("test.txt", "line1\nline2"))

    def test_create_file_nested_quotes(self):
        # Simulating: create_file('script.py', 'print("hello")')
        output = """ACTION: create_file('script.py', 'print("hello")')"""
        action, args = parse_action(output)
        self.assertEqual(action, "create_file")
        self.assertEqual(args, ("script.py", 'print("hello")'))

    def test_search_web(self):
        output = "ACTION: search_web('python regex')"
        action, args = parse_action(output)
        self.assertEqual(action, "search_web")
        self.assertEqual(args, "python regex")

    def test_negative_search(self):
        output = "ACTION: negative_search('exclude this')"
        action, args = parse_action(output)
        self.assertEqual(action, "negative_search")
        self.assertEqual(args, "exclude this")

    def test_stop(self):
        output = "ACTION: stop"
        action, args = parse_action(output)
        self.assertEqual(action, "stop")
        self.assertIsNone(args)

    def test_no_action(self):
        output = "Just thinking about stuff."
        action, args = parse_action(output)
        self.assertIn(action, ["reasoning", "unknown"])

    def test_complex_content_with_parens(self):
        # Content containing ) should not break parsing
        output = "ACTION: create_file('test.txt', 'some content (with parens)')"
        action, args = parse_action(output)
        self.assertEqual(action, "create_file")
        self.assertEqual(args, ("test.txt", "some content (with parens)"))

    def test_mixed_quote_types(self):
        # Support double quotes if implemented, though prompt mainly implies single quotes pattern.
        # But robust parsing should probably handle it.
        output = 'ACTION: run_shell("ls -la")'
        action, args = parse_action(output)
        # If we support double quotes:
        if action == "run_shell":
             self.assertEqual(args, "ls -la")

if __name__ == '__main__':
    unittest.main()
