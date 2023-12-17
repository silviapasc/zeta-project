from ken_project.zeta import (set_cwd, read_text, define_dictionary,
                              create_df, lowercase, lowercase_corpus,
                              tokenize, tokenize_corpus, build_segments,
                              build_segments_corpus, feature_occurs, feature_occurs_corpus,
                              count_segments_with_feature, sort_texts_descending, remove_stopwords,
                              remove_stopwords_corpus, segments_count)
import pytest
import os, re
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


# Test case for a non-empty corpus_dict
def test_create_df_non_empty_dict():
    # Define a dictionary
    corpus_dict = {'file1.txt': 'This is file #1.', 'file2.txt': 'This is file #2.'}
    # Define a pandas dataframe from the above dictionary by means of the function under test
    df = create_df(corpus_dict)
    # Assert the basic characteristics for a non-empty dataframe
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.shape[0] == len(corpus_dict)
    assert df.columns.tolist() == ['File Name', 'Text']
    assert df['File Name'].tolist() == list(corpus_dict.keys())
    assert df['Text'].tolist() == list(corpus_dict.values())


# Test case for an empty corpus_dict
def test_create_df_empty_dict():
    # Define an empty dictionary
    corpus_dict = {}
    # Define a pandas dataframe from the above dictionary by means of the function under test
    df = create_df(corpus_dict)
    # Assert the basic characteristics for an empty dataframe
    assert isinstance(df, pd.DataFrame)
    assert df.empty


# Test case for a corpus_dict with duplicate file names
def test_create_df_duplicate_file_names():
    corpus_dict = {'file1.txt': 'This is file #1.', 'file1.txt': 'This is a duplicate file #1.'}
    with pytest.raises(Exception) as e:
        # In the block, call the function that is expected to fail
        create_df(corpus_dict)
        assert e.type == ValueError


# Test case for lowercase() function
def test_lowercase():
    # Assign a partial uppercase text to the upper_text variable
    upper_text = 'Change this Text to LÖWERCASE'
    assert lowercase(upper_text) == 'change this text to löwercase'


# Test case for lowercase_corpus() function
def test_lowercase_corpus():
    # Define a list with partial uppercase text
    upper_list = ['First UPPER TEXT', '2ND UppeR TexT']
    assert lowercase_corpus(upper_list) == ['first upper text', '2nd upper text']


def test_tokenize():
    # The 're' module has to be imported for this test case
    # Define some string texts and assign them to a variable
    text1 = "Hello, world!"
    text2 = "This is a test @123.com http//test.com"
    text3 = "Can't you get'em all?"

    expected_tokens1 = ["Hello", "world"]
    expected_tokens2 = ["This", "is", "a", "test", "123com", "httptestcom"]
    expected_tokens3 = ["Cant", "you", "getem", "all"]

    assert tokenize(text1) == expected_tokens1
    assert tokenize(text2) == expected_tokens2
    assert tokenize(text3) == expected_tokens3


def test_tokenize_corpus():
    # Define a list of texts to be split into tokens
    texts_list = ["This is a test.", "Another @my-test.com test!"]
    # Note that no punctuation is included because of test_tokenize()
    expected_result = [["This", "is", "a", "test"], ["Another", "mytestcom", "test"]]
    assert tokenize_corpus(texts_list) == expected_result


def test_remove_stopwords():
    stopwords = ['a', 'and', 'is']
    tokenized_text = ['this', 'is', 'a', 'test']
    expected_result = ['this', 'test']
    assert remove_stopwords(stopwords, tokenized_text) == expected_result


def test_remove_stopwords_corpus():
    stopwords = ['a', 'and', 'is']
    tokens_col = [['this', 'is', 'a', 'test'], ['and', 'this', 'is', 'another', 'test']]
    expected_result = [['this', 'test'], ['this', 'another', 'test']]
    assert remove_stopwords_corpus(stopwords, tokens_col) == expected_result


def test_build_segments():
    tokens = ["A", "first", "segment", ".", "Here", "is", "a", "second", "."]
    segment_length = 3
    expected_result = [["A", "first", "segment"], [".", "Here", "is"], ["a", "second", "."]]
    assert build_segments(tokens, segment_length) == expected_result


def test_build_segments_corpus():
    tokens_lists = [["The", "first", "text", "ends", "."], ["The", "second", "one", "too", "."]]
    segment_len = 3
    expected_result = [[["The", "first", "text"], ["ends", "."]], [["The", "second", "one"], ["too", "."]]]
    assert build_segments_corpus(tokens_lists, segment_len) == expected_result


def test_segments_count():
    segments_col = pd.Series([[["The", "first", "text"], ["ends", "."]], [["The", "second"], ["one", "too"], ["."]]])
    expected_result = pd.Series([2, 3])
    # Check that left and right Series are equal
    pd.testing.assert_series_equal(segments_count(segments_col), expected_result)


def test_feature_occurs():
    segment_list = [["This", "is", "a"], ["test", "."], ["Another", "test", "sentence"], ["with", "feature"]]
    feature = "test"
    expected_result = [["test", "."], ["Another", "test", "sentence"]]
    assert feature_occurs(segment_list, feature) == expected_result


# Basically this is the same function as above, just applied to all samples (i.e. lists
# of segments) from a dataframe column
def test_feature_occurs_corpus():
    segment_list = [[["Find", "a", "hashtag"], ["here", "."]], [["Another", "test", "sentence"], ["with", "hashtag"]]]
    feature = "hashtag"
    expected_result = [[["Find", "a", "hashtag"]], [["with", "hashtag"]]]
    assert feature_occurs_corpus(segment_list, feature) == expected_result


def test_count_segments_with_feature():
    segments = [[["Find", "a", "hashtag"], ["Next", "hashtag"]], [["with", "hashtag"]]]
    segments_count = [2, 1]
    assert count_segments_with_feature(segments) == segments_count


# Create a test dataframe for this test case
@pytest.fixture
def test_dataframe():
    data = {
        'Text': ['This is a text example.', 'Each text is related to a value.', 'The text with the highest value is '
                                                                                'displayed at the top.'],
        'Value': [3, 1, 5]
    }
    return pd.DataFrame(data)


def test_sort_texts_descending(test_dataframe):
    column = 'Value'
    expected_df = pd.DataFrame({
        'Text': ['This is a text example.', 'Each text is related to a value.', 'The text with the highest value is '
                                                                                'displayed at the top.'],
        'Value': [3, 1, 5]
    })
    expected_df = expected_df.sort_values(by=column, ascending=False)
    result = sort_texts_descending(test_dataframe, column)
    pd.testing.assert_frame_equal(result, expected_df)
