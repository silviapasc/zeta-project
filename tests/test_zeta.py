import os

import pandas as pd
import pytest

from zeta_project.zeta import (set_cwd, read_text, define_dictionary,
                               create_df, lowercase, lowercase_corpus,
                               tokenize, tokenize_corpus, lemmata_pos_ner_tag,
                               build_segments, build_segments_corpus, feature_occurs,
                               feature_occurs_corpus, count_segments_with_feature, sort_descending,
                               remove_stopwords, remove_stopwords_corpus, replace_pattern_in_column,
                               segments_count, define_partitions, total_count, ratio, zeta, fill_dataframe)


# Test set_cwd() using the tmp_path fixture, which provides a temporary
# directory unique to the test invocation
def test_set_cwd(tmp_path):
    # Test when the specified path is equal to os.getcwd()
    current_path = os.getcwd()
    assert set_cwd(current_path) == current_path

    # Test when the specified path is different from os.getcwd()
    new_path = tmp_path
    assert set_cwd(new_path) == os.chdir(new_path)
    with pytest.raises(FileNotFoundError) as invalidPath:
        set_cwd("InvalidPath")
        assert invalidPath.value == "InvalidPath"


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


# Test case for a non-empty dataframe
def test_create_df_non_empty_dict():
    # Define a dictionary
    corpus_dict = {'file1.txt': 'This is file #1.', 'file2.txt': 'This is file #2.'}

    # Define a pandas dataframe from the above dictionary by means of the function under test
    df = create_df(corpus_dict)

    # Assert the basic characteristics for a non-empty dataframe
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.shape[0] == len(corpus_dict)
    # The column field 'idno' is internally set in create_df()
    assert df.columns.tolist() == ['idno', 'Text']
    assert df['idno'].tolist() == list(corpus_dict.keys())
    assert df['Text'].tolist() == list(corpus_dict.values())


# Test case for an empty dataframe
def test_create_df_empty_dict():
    # Define an empty dictionary
    corpus_dict = {}

    # Define a pandas dataframe from the above dictionary by means of the function under test
    df = create_df(corpus_dict)

    # Assert the basic characteristics for an empty dataframe
    assert isinstance(df, pd.DataFrame)
    assert df.empty


# Test case for lowercase() function
def test_lowercase():
    # Assign a partial uppercase text to the upper_text variable
    upper_text = 'Change this Text to LÖWERCASE'

    # Call the function under test and check if returned and expected content match
    assert lowercase(upper_text) == 'change this text to löwercase'


# Test case for lowercase_corpus() function
def test_lowercase_corpus():
    # Define a list with partial uppercase text
    upper_list = ['First UPPER TEXT', '2ND UppeR TexT']

    # Call the function under test and check if returned and expected content match
    assert lowercase_corpus(upper_list) == ['first upper text', '2nd upper text']


# Test case for test_tokenize() function
def test_tokenize():
    # The 're' module has to be imported for this test case
    # Define some string texts and assign them to a variable
    text1 = "Hello, world!"
    text2 = "This is a test @123.com http//test.com"
    text3 = "Can't you get'em all?"

    # Define expected lists of tokens for the corresponding 'text1', 'text2', 'text3'
    expected_tokens1 = ["Hello", "world"]
    expected_tokens2 = ["This", "is", "a", "test", "123com", "httptestcom"]
    expected_tokens3 = ["Cant", "you", "getem", "all"]

    # Call the function under test and check if returned and expected content match
    assert tokenize(text1) == expected_tokens1
    assert tokenize(text2) == expected_tokens2
    assert tokenize(text3) == expected_tokens3


# Test case for tokenize_corpus() function
def test_tokenize_corpus():
    # Define a list of texts to be split into tokens
    texts_list = ["This is a test.", "Another @my-test.com test!"]

    # Note that no punctuation is included because of test_tokenize()
    expected_result = [["This", "is", "a", "test"], ["Another", "mytestcom", "test"]]

    # Call the function under test and check if returned and expected content match
    assert tokenize_corpus(texts_list) == expected_result


# Test case for lemmata_pos_ner_tag() function
def test_lemmata_pos_ner_tag():
    # Define a pandas series containing some string texts
    texts_col = pd.Series(["Mr. Hungerton, her father, really was", "the most tactless person upon earth,"])

    # Define a list of corresponding lemmata, POS and NER tags for each text in the pandas series
    expected_lemma = [['Mr.', 'Hungerton', ',', 'her', 'father', ',', 'really', 'be'], ['the', 'most', 'tactless', 'person', 'upon', 'earth', ',']]
    expected_pos = [['PROPN', 'PROPN', 'PUNCT', 'PRON', 'NOUN', 'PUNCT', 'ADV', 'AUX'], ['DET', 'ADV', 'ADJ', 'NOUN', 'SCONJ', 'NOUN', 'PUNCT']]
    expected_ner = [['PERSON'], []]

    # Call the function under test and assign its results to separate variables
    actual_lemma, actual_pos, actual_ner = lemmata_pos_ner_tag(texts_col)

    # Check if returned and expected content match
    assert actual_lemma == expected_lemma
    assert actual_pos == expected_pos
    assert actual_ner == expected_ner


# Test case for remove_stopwords() function
def test_remove_stopwords():
    # Define a list of stopwords
    stopwords = ['a', 'and', 'is']

    # Define a list of string tokens
    tokenized_text = ['this', 'is', 'a', 'test']

    # Define a list of tokens which are no stopwords
    expected_result = ['this', 'test']

    # Call the function under test and check if returned and expected content match
    assert remove_stopwords(stopwords, tokenized_text) == expected_result


# Test case for remove_stopwords_corpus() function
def test_remove_stopwords_corpus():
    # Define a list of stopwords
    stopwords = ['a', 'and', 'is']

    # Define a list of string tokens lists
    tokens_col = [['this', 'is', 'a', 'test'], ['and', 'this', 'is', 'another', 'test']]

    # Define a list of all tokens lists, that do not include stopwords
    expected_result = [['this', 'test'], ['this', 'another', 'test']]

    # Call the function under test and check if returned and expected content match
    assert remove_stopwords_corpus(stopwords, tokens_col) == expected_result


# Test case for replace_pattern_in_column() function
def test_replace_pattern_in_column():
    # Define a pandas series containing a list of strings
    series = pd.Series(['text1.txt', 'text2.pdf', 'text3.txt'])

    # Call the function under test and save the result into a variable
    result = replace_pattern_in_column(series, '.txt', '')

    # Assert that returned and expected content match
    assert result.equals(pd.Series(['text1', 'text2.pdf', 'text3']))


# Create a test dataframe for test_define_partitions()
@pytest.fixture
def test_dataframe():
    data = {'Text': ['Text 1', 'Text 2', 'Text 3', 'Text 4', 'Text 5'],
            'Value': ['a', 'b', 'a', 'c', 'a']}
    return pd.DataFrame(data)


# Test case for define_partitions() function
def test_define_partitions(test_dataframe):
    # Call the function under test and save the result into a variables pair
    target, reference = define_partitions(test_dataframe, 'Value', 'a')

    # Assert that returned and expected content match
    assert target.equals(
        pd.DataFrame({'Text': ['Text 1', 'Text 3', 'Text 5'], 'Value': ['a', 'a', 'a']}, index=[0, 2, 4]))
    assert reference.equals(pd.DataFrame({'Text': ['Text 2', 'Text 4'], 'Value': ['b', 'c']}, index=[1, 3]))


# Test case for build_segments() function
def test_build_segments():
    # Create a list of string tokens
    tokens = ["A", "first", "segment", ".", "Here", "is", "a", "second", "."]

    # Set the parameter 'segment_length'
    segment_length = 3

    # Save the function expected result into a variable
    expected_result = [["A", "first", "segment"], [".", "Here", "is"], ["a", "second", "."]]

    # Call the function under test and assert that returned and expected content match
    assert build_segments(tokens, segment_length) == expected_result
    with pytest.raises(ValueError) as NoZeroValue:
        build_segments(tokens, 0)
        assert NoZeroValue.value == "Segment length cannot be zero"


# Test case for build_segments_corpus() function
def test_build_segments_corpus():
    # Create a list of string tokens lists
    tokens_lists = [["The", "first", "text", "ends", "."], ["The", "second", "one", "too", "."]]

    # Set the parameter 'segment_length'
    segment_len = 3

    # Save the function expected result into a variable
    expected_result = [[["The", "first", "text"], ["ends", "."]], [["The", "second", "one"], ["too", "."]]]

    # Call the function under test and assert that returned and expected content match
    assert build_segments_corpus(tokens_lists, segment_len) == expected_result
    with pytest.raises(ValueError) as NoZeroValue:
        build_segments(tokens_lists, 0)
        assert NoZeroValue.value == "Segment length cannot be zero"


# Test case for segments_count() function
def test_segments_count():
    # Define pandas series containing a list of string sublists and a list of integer values
    segments_col = pd.Series([[["The", "first", "text"], ["ends", "."]], [["The", "second"], ["one", "too"], ["."]]])
    expected_result = pd.Series([2, 3])

    # Assert that the result of the function under test and the 'expected_result' variable match
    pd.testing.assert_series_equal(segments_count(segments_col), expected_result)


# Test case for total_count() function
def test_total_count():
    # Define a pandas series of integer values
    single_counts = pd.Series([1, 2, 3, 4, 5])

    # Save the sum of integer values into a variable
    expected_result = 1 + 2 + 3 + 4 + 5

    # Call the function under test and check if returned and expected content match
    assert total_count(single_counts) == expected_result
    with pytest.raises(TypeError) as NoFloat:
        total_count(pd.Series([0.1, 0.2, 3, 4, 5]))
        assert NoFloat.type == "Floating point values not allowed"


# Test case for feature_occurs() function
def test_feature_occurs():
    # Define a list of tokens segments
    segment_list = [["This", "is", "a"], ["test", "."], ["Another", "test", "sentence"], ["with", "chosen_feature"]]

    # Define a feature to be found within the segments and save it into a variable
    feature = "test"

    # Define a list of expected tokens segments
    expected_result = [["test", "."], ["Another", "test", "sentence"]]

    # Call the function under test and check if returned and expected content match
    assert feature_occurs(segment_list, feature) == expected_result


# Test case for feature_occurs_corpus() function
# Basically this is the same function as the one described above, but applied to all samples (i.e. lists
# of segments) from a dataframe column
def test_feature_occurs_corpus():
    segment_list = [[["Find", "a", "hashtag"], ["here", "."]], [["Another", "test", "sentence"], ["with", "hashtag"]]]
    feature = "hashtag"
    expected_result = [[["Find", "a", "hashtag"]], [["with", "hashtag"]]]
    assert feature_occurs_corpus(segment_list, feature) == expected_result


# Test case for count_segments_with_feature() function
def test_count_segments_with_feature():
    # Define a list of nested tokens segments
    segments = [[["Find", "a", "hashtag"], ["Next", "hashtag"]], [["with", "hashtag"]]]

    # Define a list of expected integer values as counts
    segments_count = [2, 1]

    # Call the function under test and check if returned and expected content match
    assert count_segments_with_feature(segments) == segments_count


# Create a test dataframe for this test case
@pytest.fixture
def test_dataframe1():
    data = {
        'Text': ['This is a text example.', 'Each text is related to a value.', 'The text with the highest value is '
                                                                                'displayed at the top.'],
        'Value': [3, 1, 5]
    }
    return pd.DataFrame(data)


# Test case for sort_descending() function
def test_sort_descending(test_dataframe1):
    # Instantiate the column field 'column' and a dataframe 'expected_df'
    column = 'Value'
    expected_df = pd.DataFrame({
        'Text': ['This is a text example.', 'Each text is related to a value.', 'The text with the highest value is '
                                                                                'displayed at the top.'],
        'Value': [3, 1, 5]
    })

    # Sort the 'expected_df' dataframe values by the column 'Value' in descending order
    expected_df = expected_df.sort_values(by='Value', ascending=False)

    # Call the function under test for 'test_dataframe1' and save the result into a variable
    result = sort_descending(test_dataframe1, column)

    # Assert that the result of the function under test and the 'expected_df' variable match
    pd.testing.assert_frame_equal(result, expected_df)


# Test case for ratio() function
def test_ratio():
    # Assign integer values to the variables 'value_1' and 'value_2'
    value_1 = 27
    value_2 = 85

    # Compute the ratio between the same integer values
    expected_ratio = 27 / 85

    # Call the function under test and check if returned and expected content match
    assert ratio(value_1, value_2) == expected_ratio


# Test case for zeta() function
def test_zeta():
    # Assign float values to the variables 'ratio_1' and 'ratio_2'
    ratio_1 = 0.32
    ratio_2 = 0.75

    # Compute the difference between the same integer values
    expected_zeta = 0.32 - 0.75

    # Call the function under test and check if returned and expected content match
    assert zeta(ratio_1, ratio_2) == expected_zeta


# Test case for fill_dataframe() function
def test_fill_dataframe():
    # Define an empty dataframe
    df = pd.DataFrame(columns=['A', 'B', 'C'])

    # Add values
    values = [1, 'string', 3]

    # Call the function under test and save the result into a variable
    df_updated = fill_dataframe(df, values)

    # Assert if returned and expected content match
    assert df_updated.equals(pd.DataFrame({'A': [1], 'B': ['string'], 'C': [3]}))
