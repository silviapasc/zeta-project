# Import the required modules
import os
import re
import pandas as pd


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


def read_text(filename: str) -> bytes:
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


# Remove stopwords with a specific function!


# Tokenize lowercase corpus
def tokenize_corpus(texts_col: list) -> list:
    """ Tokenizes the corpus texts within the 'Text' column
    returning a list of token lists. Punctuation is also removed """
    return [tokenize(file) for file in texts_col]


# Set a function to build the text segments (ideally 2000-5000 tokens)
def build_segments(tokens: list, segment_length: int) -> list:
    """ Builds a series of token sublists or segments based on the given segment length.
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


# Consider now these subsets of tokens as the unit to check if the chosen feature
# occurs within the texts. If the feature occurs at least once, this is what matters
def feature_occurs(segments_list: list, feature: str) -> list:
    """ Returns a list of those segments containing the specified feature """
    result = [segment for segment in segments_list if feature in segment]
    return result


def feature_occurs_corpus(segments_column: list, feature: str) -> list:
    """ Returns a list, for each dataframe sample, of only those segments
    containing the specified feature """
    return [feature_occurs(segments, feature) for segments in segments_column]


# After that, count in how many segments (of the partition) the merkmal occurs, i.e. totally
# 'segment_list' can be a list of lists or tuples

# Count the number of segments containing the feature for each text within the corpus
# Gives a tuple back with integer value of segments containing feature
# and a list of the segments themselves
# There is a conflict between the total index number of the dataframe
# and the number of results, because for a single index there can be
# more results and for some other there is no results


# Count the total number of segments containing the feature for each document,
# i.e. dataframe row. Create a new column for that?
def count_segments_with_feature(segments_column):
    return [len(segments) for segments in segments_column]


# return [segments for segments in container if feature in segments]
# result = feature_occurs(container, feature)

### segments_with_features_total = df['Number of Segments with Feature Occurrence'].sum()
### df['Total Segments'] = df['Segments'].apply(len)
### segments_total = df['Total Segments'].sum()


# Within the two partitions sort the texts so that
# the text with the highest number of segments containing the feature on top
# and the text with the lowest number of segments containing the feature at the bottom
# It would be better to create a mask (i.e. dataframe reduced) and
# output only the text index/title and the relative counts
def sort_texts_descending(dataframe, column):
    return dataframe.sort_values(by=column, ascending=False)


# Now you should think about how to compare the two partitions
# (text-and-counts vs text-and-counts; maybe merge p1 and p2)
# and before that about the parameter to set them

# Workflow
# Set a parameter to separate the two partitions;
# Process the texts within each subset;
# Get the number of sequences containing the chosen feature;
# Sort the two subsets in descending order (texts with more sequences-with-feature
# on the top and texts with less sequences-with-feature at the bottom)

# Printing the output of the functions
if __name__ == '__main__':
    # this will we deleted when the script is complete
    print('Executing as standalone script')
    print(f'This is the the output of the function: insert a function here')

    path = input("Enter the path to the text corpus/partition: ")
    dictionary = define_dictionary(path)
    df = create_df(dictionary)
    df['Lowercase Text'] = lowercase_corpus(df.Text)
    df['Tokenized Text'] = tokenize_corpus(df['Lowercase Text'])
    df['Segments'] = build_segments_corpus(df['Tokenized Text'], 1000)
    # print(type(df['Tokenized Text']))
