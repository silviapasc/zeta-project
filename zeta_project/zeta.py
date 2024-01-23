# Import the required modules
import os
import re

import pandas as pd
from pandas import DataFrame
import spacy


# Set the proper working directory path
def set_cwd(current_path: str) -> str:
    """ The function checks whether the specified path matches the
    current working directory path. If yes, the specified path is returned.
    Otherwise, the working directory is changed to the specified path."""
    if current_path == os.getcwd():
        return current_path
    else:
        return os.chdir(current_path)

    # What if an invalid directory path is selected? Actually a
    # FileNotFoundError is raised by the following function read_corpus()


def read_text(filename: str) -> str:
    """ Reads the text file content, returning the specified number of bytes.
    Default -1, which means the whole file"""
    # How to explain str instead of bytes???
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


# Define a pandas dataframe from dictionary
def create_df(corpus_dict: dict) -> pd.DataFrame:
    """ Creates a pandas dataframe from a dictionary defined by define_dictionary().
     The dictionary keys are collected under the 'File Name' column and the values
     under the 'Text' column."""
    # add some if-raise-conditions? Consider exceptions
    return pd.DataFrame(corpus_dict.items(), columns=['idno', 'Text'])


def lowercase(text: str) -> str:
    """ Converts strings to lowercase."""
    return text.lower()


def lowercase_corpus(texts_col: list) -> list:
    """ Converts the corpus texts within the 'Text' column to lowercase."""
    return [lowercase(file) for file in texts_col]


def tokenize(text: str) -> list:
    """ Tokenizes a string text returning a list of tokens.
    Uses the 're' module to remove punctuation."""
    new_text = re.sub(r'[^\w\s]', '', text)
    tokens = new_text.split()
    return tokens


# Tokenize lowercase corpus
def tokenize_corpus(texts_col: list) -> list:
    """ Tokenizes the string texts within the 'Text' column
    returning a list of token lists. Punctuation is also removed."""
    return [tokenize(file) for file in texts_col]


# Extract lemmata, Part-Of-Speech and Named-Entity-Recognition tags
# from the string texts using the Spacy library
def lemmata_pos_ner_tag(texts_col: pd.Series) -> list:
    """ Tokenizes the string texts within a pandas series and returns lemmata,
    Part-Of-Speech (POS) and Named-Entity-Recognition (NER) tags within
    separate lists. The functionality is based on the SpaCy library, which
    has to be imported before, and relies on the specific model 'en_core_web_sm'."""
    nlp = spacy.load("en_core_web_sm")
    lemma = []
    pos = []
    ner = []
    for doc in nlp.pipe(texts_col, batch_size=20):
        lemma.append([token.lemma_ for token in doc])
        pos.append([token2.pos_ for token2 in doc])
        ner.append([token3.label_ for token3 in doc.ents])
    return lemma, pos, ner


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


def replace_pattern_in_column(column: pd.Series, old_pattern: str, new_pattern: str) -> pd.Series:
    """ Replaces each matching string pattern from dataframe column values with a new
     pattern, which can be also an empty string. Returns the updated dataframe column"""
    return column.str.replace(old_pattern, new_pattern, regex=True)


# Split the dataframe into 2 partitions based on a chosen value from a selected column
def define_partitions(dataframe: pd.DataFrame, col_name: str, col_value: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Splits a dataframe into two partitions, one referred to as the 'target partition' and the other
     as the 'reference partition'. The split is based on the specified dataframe column and a column value,
     in a way that all the dataframe rows showing that column value are grouped into the target partition,
     while the remaining rows are grouped into the reference partition. """
    split_condition = dataframe[col_name] == col_value
    # Explain the meaning of the copy() method!
    target_partition = dataframe[split_condition].copy()
    reference_partition = dataframe[~split_condition].copy()
    return target_partition, reference_partition


# Set a function to build the text segments (ideally 2000-5000 tokens)
def build_segments(tokens: list, segment_len: int) -> list:
    """ Builds a series of token sub lists or segments based on the given segment length.
    The segment length corresponds to the number of tokens, each segment is made of.
    """
    # The index at which the tokens should be split is defined through the slice syntax.
    # The value '0' corresponds to the starting point, the total number of tokens – len(tokens) –
    # marks the stop position and the specified number of tokens sets the interval at which
    # the split occurs
    return [tokens[x: x + segment_len] for x in range(0, len(tokens), segment_len)]


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


# Consider now these subsets of tokens or segments as the unit to check if the specified feature
# occurs within the texts. If the chosen feature occurs at least once, this is what matters
def feature_occurs(segments_list: list, feature: str) -> list:
    """ Returns a list of those segments containing the specified feature """
    result = [segment for segment in segments_list if feature in segment]
    return result


def feature_occurs_corpus(segments_column: list, feature: str) -> list:
    """ Returns a list, for each dataframe sample, of only those segments
    containing the specified feature """
    return [feature_occurs(segments, feature) for segments in segments_column]


# Count the total number of segments containing the chosen feature for each document
def count_segments_with_feature(segments_column: list) -> list[int]:
    """ Returns a list with the total number of segments containing the specified
    feature. The total number is referred to each text item """
    return [len(segments) for segments in segments_column]


# Sum the number of segments containing the specified feature for the whole corpus partition
def total_count(column_counts: pd.Series) -> int:
    """ Returns the sum of a series of integer values """
    return column_counts.sum()


# Compute the ratio of the total number of segments containing the feature
# over the total number of segments within a partition
def ratio(segments_with_feature_count: int, segments_count: int) -> float:
    """ Returns the percentage of segments containing the specified feature
    over all segments of a partition"""
    return segments_with_feature_count / segments_count


# Compute zeta
def zeta(ratio_1: float, ratio_2: float) -> float:
    """ Returns the percentage of how consistently the specified feature
    is used within the target partition compared to the reference partition"""
    return ratio_1 - ratio_2


# Insert a list of values into a dataframe
def fill_dataframe(dataframe: pd.DataFrame, values: list) -> pd.DataFrame:
    """ Inserts the specified list of values into the existing dataframe. The number of values
    must correspond to the number of labels in the dataframe"""
    dataframe.loc[len(dataframe.index)] = values
    return dataframe


# Sort the samples within the dataframe/partition, so that the text item with
# the highest number of segments containing the specified feature figures at the top,
# and the text with the lowest number of segments containing the feature is
# at the bottom
def sort_descending(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    """ Returns a dataframe sorted by the values from the specified column, in descending order """
    return dataframe.sort_values(by=column, ascending=False)


# Executing as standalone script
if __name__ == '__main__':
    # First create a dataframe to which append the results
    # Module 'DataFrame' from pandas is required!
    summary: DataFrame = pd.DataFrame(
        columns=['Feature', 'Target Partition Ratio', 'Reference Partition Ratio', 'Zeta Value'])

    corpus_path = input("Enter the directory path to the text corpus: ")
    dictionary = define_dictionary(corpus_path)
    df = create_df(dictionary)
    # Extra function to remove file extension suffix and make it match with the metadata table value
    df['idno'] = replace_pattern_in_column(df['idno'], '.txt$', '')

    # Preprocess all the corpus texts
    df['Lowercase Text'] = lowercase_corpus(df.Text)
    df['Tokenized Text'] = tokenize_corpus(df['Lowercase Text'])
    # Define lemmata, Part-of-Speech-tags and Named-Entity-Recognition-tags
    lemmata_and_pos = lemmata_pos_ner_tag(df["Text"])
    df["Lemmata"] = lemmata_and_pos[0]
    df["POS"] = lemmata_and_pos[1]
    df["NER"] = lemmata_and_pos[2]
    # df["POS"] = corpus_pos(df["Text"])
    # df['Lemmata'] = tokenize_corpus(df.Text)
    # df['PartOfSpeech'] = tokenize_corpus(df.Text)
    # Remove stopwords if necessary
    # stopwords = input("Add a list of stopwords (empty as default value): ")
    # df['Text No Stopwords'] = remove_stopwords_corpus(list(stopwords), df['Tokenized Text'])
    # Set segments length
    segment_length = input("Specify the desired segment length (in tokens): ")
    # Instead of df['Tokenized Text'], use df["Lemmata"], df["POS"] or df['NER']
    # With df["POS"] you should know the list of possible tags, e.g.
    df['Segments'] = build_segments_corpus(df['NER'], int(segment_length))
    # df['Segments'] = build_segments_corpus(df['Text No Stopwords'], int(segment_length))
    df['Segments Count'] = segments_count(df['Segments'])
    print(df)

    # Get metadata and read it to a dataframe
    meta_path = input("Enter the directory path to the metadata: ")
    meta = pd.read_csv(meta_path, sep='\t', encoding='UTF-8')

    # Merge text corpus dataframe with metadata dataframe
    merged = df.merge(meta, how='left', on='idno')
    print(merged)

    # Split merged dataframe into target and reference partition based on metadata values
    meta_col, meta_value = [item for item in input("Specify the metadata column name and a corresponding value to "
                                                   "split the corpus into target and reference partition\n (use shift "
                                                   "key to separate): ").split()]
    zp, vp = define_partitions(merged, meta_col, meta_value)

    while True:
        # Specify a feature with respect to which calculate zeta
        chosen_feature = input("Specify a feature: ")

        # Process data within target partition
        zp['Feature Occurrence'] = feature_occurs_corpus(zp['Segments'], chosen_feature)
        zp['Number of Segments with Feature'] = count_segments_with_feature(zp['Feature Occurrence'])
        zp_sorted = sort_descending(zp, 'Number of Segments with Feature')

        # Process data within reference partition
        vp['Feature Occurrence'] = feature_occurs_corpus(vp['Segments'], chosen_feature)
        vp['Number of Segments with Feature'] = count_segments_with_feature(vp['Feature Occurrence'])
        vp_sorted = sort_descending(vp, 'Number of Segments with Feature')

        # Target partition and reference partition dataframes output
        print(zp_sorted)
        print(vp_sorted)

        # Count total segments, segments containing the feature and ratio of that within the target partition
        total_segments = total_count(zp_sorted['Segments Count'])
        total_segments_with_features = total_count(zp_sorted['Number of Segments with Feature'])
        print(f'Total of segments within the target partition: ', total_segments)
        print(f'Total of segments with chosen_feature within the target partition: ', total_segments_with_features)
        zp_ratio = ratio(total_segments_with_features, total_segments)
        print(f'Ratio: ', zp_ratio)

        # Count total segments, segments containing the feature and ratio of that within the reference partition
        total_segments = total_count(vp_sorted['Segments Count'])
        total_segments_with_features = total_count(vp_sorted['Number of Segments with Feature'])
        print(f'Total of segments within the target partition: ', total_segments)
        print(f'Total of segments with chosen_feature within the target partition: ', total_segments_with_features)
        vp_ratio = ratio(total_segments_with_features, total_segments)
        print(f'Ratio: ', vp_ratio)

        # Calculate Zeta, with values range [-1,1]
        zeta_value = zeta(zp_ratio, vp_ratio)
        print(f'Zeta value with reference to the chosen feature "', chosen_feature, '" : ', zeta_value)
        fill_dataframe(summary, [chosen_feature, zp_ratio, vp_ratio, zeta_value])

        new_feature = input("Any other feature? (y/n): ")
        if new_feature.lower() != "y":
            break

    # Sort the results dataframe by descending values of zeta
    summary = sort_descending(summary, 'Zeta Value')
    # Eventually save the definitive dataframe to a csv file in the current working directory
    # summary.to_csv('zeta-summary.csv')
    print(summary)
