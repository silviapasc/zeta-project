# Import the required modules
import os
import re

import pandas as pd
from pandas import DataFrame


# Set the proper working directory path
def set_cwd(current_path: str) -> str:
    """ The function checks whether the specified path matches the
    current working directory path. If yes, the specified path is returned.
    Otherwise, the working directory is changed to the specified path.

    Parameters:
    current_path (str): The desired directory path to which the working directory
    is eventually to be changed.

    Return value:
    str: The current working directory path if it matches the specified path.
    Otherwise, the changed working directory path is returned.

    Note:
    The function uses the 'os' module, which must be imported for the function to work properly.
    """
    if current_path == os.getcwd():
        return current_path
    else:
        return os.chdir(current_path)

    # What if an invalid directory path is selected? Actually a
    # FileNotFoundError is raised by the following function read_corpus()


def read_text(filename: str) -> str:
    """ Reads the text file content, returning the specified number of bytes.
    Default -1, which means the whole file"""
    with open(filename, 'rt', encoding='utf-8') as file:
        return file.read()


# Create a dictionary where text file names are the keys and each text content is the value.
# In this way each text file is bound to its title/index, when processed later
def define_dictionary(specified_path: str) -> dict:
    """ Creates a dictionary with the text items from the collection within
    the specified directory path. The dictionary keys correspond to the text
    file names and the values correspond to the text file contents."""
    set_cwd(specified_path)
    return {file: read_text(file) for file in os.listdir() if file.endswith(".txt")}


# Define a Pandas dataframe from dictionary
def create_df(corpus_dict: dict) -> pd.DataFrame:
    """ Creates a pandas dataframe from a dictionary defined by define_dictionary().
     The dictionary keys are collected under the 'File Name' column and the values
     under the 'Text' column."""
    # add some if-raise-conditions? Consider exceptions
    return pd.DataFrame(corpus_dict.items(), columns=['File Name', 'Text'])


def lowercase(text: str) -> str:
    """ Converts strings to lowercase."""
    return text.lower()


def lowercase_corpus(texts_col: list) -> list:
    """ Converts the corpus texts within the 'Text' column to lowercase."""
    return [lowercase(file) for file in texts_col]


def tokenize(text: str) -> list:
    """ Tokenizes a string text returning a list of tokens.
    Uses the 're' module to remove punctuation"""
    new_text = re.sub(r'[^\w\s]', '', text)
    tokens = new_text.split()
    return tokens


# Tokenize lowercase corpus
def tokenize_corpus(texts_col: list) -> list:
    """ Tokenizes the corpus texts within the 'Text' column
    returning a list of token lists. Punctuation is also removed """
    return [tokenize(file) for file in texts_col]


# Eventually remove stopwords
def remove_stopwords(stopwords_list: list, tokenized_text: list) -> list:
    """ Removes stopwords from a list of string tokens, returning
    a list of the filtered tokens. The stopwords themselves are
    contained as string tokens within a list"""
    return [token for token in tokenized_text if token not in stopwords_list]


def remove_stopwords_corpus(stopwords_list: list, tokens_col: list) -> list:
    """ Removes stopwords from each string tokens list within a dataframe.
    The stopwords themselves are contained as string tokens within a list"""
    return [remove_stopwords(stopwords_list, tokenized_text) for tokenized_text in tokens_col]


# Set a function to build the text segments (ideally 2000-5000 tokens)
def build_segments(tokens: list, segment_length: int) -> list:
    """ Builds a series of token sub lists or segments based on the given segment length.
    The segment length corresponds to the number of tokens, each segment is made of.
    """
    # The index at which the tokens should be split is defined through the slice syntax.
    # The value '0' corresponds to the starting point, the total number of tokens – len(tokens) –
    # marks the stop position and the specified number of tokens sets the interval at which
    # the split occurs
    return [tokens[x: x + segment_length] for x in range(0, len(tokens), segment_length)]


# Build segments for each text of the corpus
def build_segments_corpus(tokens_lists: list, segment_len: int) -> list:
    """ Builds a list of token segments from each tokens list, in which
    the corpus texts are split into. The argument 'segment_len' specifies
    the number of tokens to be included in each segment """
    return [build_segments(tokenized, segment_len) for tokenized in tokens_lists]


# Count total number of segments for each text
def segments_count(segments_col: pd.Series) -> pd.Series:
    """ Returns a series of integers from a series of string
    tokens segments. The integers correspond to the total number
    of segments in each text of the collection. """
    return segments_col.apply(len)


# Consider now these subsets of tokens as the unit to check if the chosen chosen_feature
# occurs within the texts. If the chosen_feature occurs at least once, this is what matters
def feature_occurs(segments_list: list, feature: str) -> list:
    """ Returns a list of those segments containing the specified chosen_feature """
    result = [segment for segment in segments_list if feature in segment]
    return result


def feature_occurs_corpus(segments_column: list, feature: str) -> list:
    """ Returns a list, for each dataframe sample, of only those segments
    containing the specified chosen_feature """
    return [feature_occurs(segments, feature) for segments in segments_column]


# After that, count in how many segments (of the partition) the chosen_feature occurs, i.e. totally
# 'segment_list' can be a list of lists or tuples

# Count the number of segments containing the chosen_feature for each text within the corpus
# Gives a tuple back with integer value of segments containing chosen_feature
# and a list of the segments themselves
# There is a conflict between the total index number of the dataframe
# and the number of results, because for a single index there can be
# more results and for some other there is no results


# Count the total number of segments containing the chosen_feature for each document
def count_segments_with_feature(segments_column: list) -> list[int]:
    """ Returns a list with the total number of segments containing the specified
    chosen_feature. The total number is referred to each text item """
    return [len(segments) for segments in segments_column]


def total_count(column_counts: pd.Series) -> int:
    """ Returns the sum of a series of integer values """
    return column_counts.sum()


# Compute the ratio of the total number of segments containing the feature
# over the total number of segments within a partition
def ratio(segments_with_feature_count: int, segments_count: int) -> float:
    """ Returns the percentage of segments containing the specified feature
    over all segments of a partition"""
    return segments_with_feature_count / segments_count


# Zeta
def zeta(ratio_1: float, ratio_2: float) -> float:
    """ Returns the percentage of how consistently the specified feature
    is used within the target partition compared to the reference partition"""
    return ratio_1 - ratio_2


def fill_dataframe(dataframe: pd.DataFrame, values: list) -> pd.DataFrame:
    """ Inserts the specified list of values into the existing dataframe. The number of values
    must correspond to the number of labels in the dataframe"""
    dataframe.loc[len(dataframe.index)] = values
    return dataframe


# Within the dataframe/partition sort the samples so that the text item with
# the highest number of segments containing the chosen_feature figures at the top,
# and the text with the lowest number of segments containing the chosen_feature is
# at the bottom
def sort_descending(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    """ Returns a dataframe sorted by the values from the specified column, in descending order """
    return dataframe.sort_values(by=column, ascending=False)


# return [segments for segments in container if chosen_feature in segments]
# result = feature_occurs(container, chosen_feature)

# total_segments_with_features = df['Number of Segments with Feature Occurrence'].sum()
# df['Total Segments'] = df['Segments'].apply(len)
# segments_total = df['Total Segments'].sum()

# Now you should think about how to compare the two partitions
# (text-and-counts vs text-and-counts; maybe merge p1 and p2)
# and before that about the parameter to set them

# Workflow
# Set a parameter to separate the two partitions;
# Process the texts within each subset;
# Get the number of sequences containing the chosen chosen_feature;
# Sort the two subsets in descending order (texts with more sequences-with-chosen_feature
# on the top and texts with less sequences-with-chosen_feature at the bottom)


# Executing as standalone script
if __name__ == '__main__':
    # Create a dataframe to which append the results
    summary: DataFrame = pd.DataFrame(columns=['Feature', 'Target Partition Ratio', 'Reference Partition Ratio', 'Zeta Value'])
    while True:
        # Define a chosen_feature with respect to which calculate zeta
        chosen_feature = input("Specify a feature: ")
        # Define a target partition
        # Get input user data by means of the input() function
        path1 = input("Enter the directory path to the target partition: ")
        dictionary1 = define_dictionary(path1)
        df = create_df(dictionary1)
        df['Lowercase Text'] = lowercase_corpus(df.Text)
        df['Tokenized Text'] = tokenize_corpus(df['Lowercase Text'])
        # Remove stopwords if necessary
        # df['Text No Stopwords'] = zt.remove_stopwords_corpus(['and', 'in'], df['Tokenized Text'])
        df['Segments'] = build_segments_corpus(df['Tokenized Text'], 1000)
        df['Segments Count'] = segments_count(df['Segments'])
        df['Feature Occurrence'] = feature_occurs_corpus(df['Segments'], chosen_feature)
        df['Number of Segments with Feature'] = count_segments_with_feature(df['Feature Occurrence'])
        df_sorted1 = sort_descending(df, 'Number of Segments with Feature')
        # df_sorted1 = df.sort_values(by='Number of Segments with Feature', ascending=False)

        # Repeat the code to define the reference partition
        path2 = input("Enter the directory path to the reference partition: ")
        dictionary2 = define_dictionary(path2)
        df = create_df(dictionary2)
        df['Lowercase Text'] = lowercase_corpus(df.Text)
        df['Tokenized Text'] = tokenize_corpus(df['Lowercase Text'])
        # Remove stopwords if necessary
        # df['Text No Stopwords'] = zt.remove_stopwords_corpus(['and', 'in'], df['Tokenized Text'])
        df['Segments'] = build_segments_corpus(df['Tokenized Text'], 1000)
        df['Segments Count'] = segments_count(df['Segments'])
        df['Feature Occurrence'] = feature_occurs_corpus(df['Segments'], 'sherlock')
        df['Number of Segments with Feature'] = count_segments_with_feature(df['Feature Occurrence'])
        df_sorted2 = sort_descending(df, 'Number of Segments with Feature')
        # df_sorted2 = df.sort_values(by='Number of Segments with Feature', ascending=False)

        # Dataframes output
        print(df_sorted1)
        print(df_sorted2)

        # Repeat for each partition
        total_segments = total_count(df_sorted1['Segments Count'])
        total_segments_with_features = total_count(df_sorted1['Number of Segments with Feature'])
        print(f'Total of segments within the target partition: ', total_segments)
        print(f'Total of segments with chosen_feature within the target partition: ', total_segments_with_features)
        ratio1 = ratio(total_segments_with_features, total_segments)
        print(f'Ratio: ', ratio1)

        total_segments = total_count(df_sorted2['Segments Count'])
        total_segments_with_features = total_count(df_sorted2['Number of Segments with Feature'])
        print(f'Total of segments within the reference partition: ', total_segments)
        print(f'Total of segments with chosen_feature within the reference partition: ', total_segments_with_features)
        ratio2 = ratio(total_segments_with_features, total_segments)
        print(f'Ratio: ', ratio2)

        # Calculate Zeta – values range [-1,1] –
        zeta_value = zeta(ratio1, ratio2)
        print(f'Zeta value with reference to the chosen feature "', chosen_feature, '" : ', zeta_value)
        fill_dataframe(summary, [chosen_feature, ratio1, ratio2, zeta_value])
        # summary.loc[len(summary.index)] = [chosen_feature, ratio1, ratio2, zeta]

        repeat = input("Any other feature? (y/n): ")
        if repeat.lower() != "y":
            break
    # Use the same sort_descending()
    # summary = summary.sort_values(by='Zeta Value', ascending=False)
    summary = sort_descending(summary, 'Zeta Value')
    # Save the definitive dataframe to a csv file
    # summary.to_csv('zeta-summary.csv')
    print(summary)
