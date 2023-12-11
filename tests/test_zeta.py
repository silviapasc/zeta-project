from ken_project.zeta import (set_cwd, read_text)
import pytest
import os
import pandas as pd


# Test set_cwd() using the tmp_path fixture, which provides a temporary
# directory unique to the test invocation
def test_set_cwd(tmp_path):
    # Test when the specified path is equal to os.getcwd()
    current_path = os.getcwd()
    assert set_cwd(current_path) == current_path

    # Test when the specified path is different from os.getcwd()
    new_path = tmp_path
    assert set_cwd(new_path) == os.chdir(new_path)


def test_read_text():
    # Define a filename and its corresponding content as variables
    filename = "test_file.txt"
    file_content = "Test file content! @ #123"

    # Create/write a test file with the given content
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(file_content)

    # Call the function and check if returned and expected content match
    assert read_text(filename) == file_content
