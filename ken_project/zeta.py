# Set required working directory
def set_cwd(current_path):
    import os
    return os.chdir(current_path)


# Read the text file -> str
def read_text(filename):
    with open(filename, 'rt', encoding='utf-8') as file:
        return file.read()


# Read the corpus of text files and returns a list of them
def read_corpus(new_path):
    # Import module
    import os
    set_cwd(new_path)
    # Iterate through all file
    return [f"{file} : {read_text(file)}" for file in os.listdir() if file.endswith(".txt")]


# Need a way to associate an index/title to each text during the manipulation!
# Create dictionary where the text file name is the key and the text strings are the value
def define_dictionary(path):
    # Import module
    import os
    # Set the cwd using the set_cwd() function
    set_cwd(path)
    return {file: read_text(file) for file in os.listdir() if file.endswith(".txt")}


# Define Pandas dataframe from dictionary


# Lowercase text (str) -> str
def lowercase(text):
    return text.lower()


# Lowercase corpus
def lowercase_corpus(texts_list):
    return [lowercase(file) for file in texts_list]


# Tokenize the text file (str) -> list
# regex to remove punctuation
# Stopwords are still included
def tokenize(text):
    import re
    # uppercase not removed, because there is another function for that purpose!
    new_text = re.sub(r'[^\w\s]', '', text)
    tokens = new_text.split()
    return tokens


# Remove stopwords with a specific function!


# Tokenize lowercase corpus
def tokenize_corpus(texts_list):
    return [tokenize(file) for file in texts_list]


# Set 2000-5000 tokens as value to build segments
# Take the list or iterator and count the number of tokens up to 2000
# Here set a limit and move on to the next segment
# take an iterator -> list
def build_segments(tokens, segment_length):
    # The index at which split the tokens iterator is defined through Slicing: 'x: x + segment_length'
    # Slicing is used to retrieve a subset of values
    # To retrieve a subset of elements, the start and stop positions need to be defined
    return [tokens[x: x + segment_length] for x in range(0, len(tokens), segment_length)]


# Build segments for each text of the corpus
def build_segments_corpus(tokens_lists, segment_len):
    return [build_segments(tokenized, segment_len) for tokenized in tokens_lists]


# Consider now these subsets of tokens as unit to check if the merkmal occurs.
# Occurs the merkmal at least once, then it is valid as check
# 'segment_list' should be also iterator?
# After that, count in how many segments (of the partition) the merkmal occurs, i.e. totally
def feature_occurs(segment_list, feature):
    result = [segment for segment in segment_list if feature in segment]
    # int for the number of segments containing the feature
    # 'result' is a list of iterators (lists/tuples, but not sets)
    return len(result), result


# Count the number of segments containing the feature for each text within the corpus
# Gives a tuple back with integer value of segments containing feature
# and a list of the segments themselves
def feature_occurs_corpus(corpus_segments, feature):
    for segments in corpus_segments:
        return feature_occurs(segments, feature)


# Within the two partitions sort the texts so that
# the text with the highest number of segments containing the feature on top
# and the text with the lowest number of segments containing the feature at the bottom


# Printing the output of the functions
if __name__ == '__main__':
    # this will we deleted when the script is complete
    print('Executing as standalone script')
    print(f'This is the the output of the function: insert a function here')
