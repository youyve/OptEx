import unittest
import subprocess
import os

class TestExamples(unittest.TestCase):
    def setUp(self):
        self.examples_dir = os.path.join(os.path.dirname(__file__), '..', 'examples')

    def test_test_optex_script(self):
        script_path = os.path.join(self.examples_dir, 'test_optex.py')
        result = subprocess.run(['python', script_path], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg=f"Script failed with stderr: {result.stderr}")

    def test_example_usage_script(self):
        script_path = os.path.join(self.examples_dir, 'minst.py')
        result = subprocess.run(['python', script_path], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg=f"Script failed with stderr: {result.stderr}")

if __name__ == '__main__':
    unittest.main()
