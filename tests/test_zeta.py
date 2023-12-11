from ken_project.zeta import (set_cwd, read_text, define_dictionary)
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

    # Call the function under test and check if returned and expected content match
    assert read_text(filename) == file_content


def test_define_dictionary(tmp_path):
    # Create some test files
    file1 = tmp_path / "file1.txt"
    file1.write_text("This is a text file")
    file2 = tmp_path / "file2.txt"
    file2.write_text("This is another text file")
    file3 = tmp_path / "file3.csv"
    file3.write_text("This is not a text file, but a csv file")

    # Call the function under test
    result = define_dictionary(tmp_path)

    # Assert the expected dictionary
    assert result == {"file1.txt": "This is a text file", "file2.txt": "This is another text file"}



