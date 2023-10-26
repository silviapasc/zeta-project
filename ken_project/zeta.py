# Check your current working directory
def get_cwd():
    import os
    cwd = os.getcwd()
    return cwd


# Set required working directory
def set_cwd(current_path):
    import os
    return os.chdir(current_path)


# Read the text file -> str
def read_text(filename):
    with open(filename, 'rt', encoding='utf-8') as file:
        return file.read()


# Read the corpus
def read_corpus(new_path):
    # Import module
    import os
    # Iterate through all file
    for file in os.listdir():
        # Check whether file is in text format or not
        if file.endswith(".txt"):
            # Set the current file path
            file_path = f"{new_path}/{file}"
            # call the read_text() function
            return read_text(file_path)




# Lowercase text (str) -> str
def lowercase(text):
    return text.lower()


# Tokenize the text file (str) -> list
# regex to remove punctuation
# Stopwords are still included
def tokenize(text):
    import re
    # uppercase not removed, because there is another function for that purpose!
    new_text = re.sub(r'[^\w\s]', '', text)
    tokens = new_text.split()
    return tokens


# Remove stopswords with a specific function


# Count the frequency for each token within the text file
# Last step was a list of strings, could be iterator
#
# refactor the code here!
def word_count(tokens):
    frequencies = {}
    for token in tokens:
        if token in frequencies:
            frequencies[token] += 1
        else:
            frequencies[token] = 1
    return frequencies


# Set 2000-5000 tokens as value to build segments
# Take the list or iterator and count the number of tokens up to 2000
# Here set a limit and move on to the next segment
# take an iterator -> list
def build_segments(tokens, segment_length):
    # The index at which split the tokens iterator is defined through Slicing: 'x: x + segment_length'
    # Slicing is used to retrieve a subset of values
    # To retrieve a subset of elements, the start and stop positions need to be defined
    return [tokens[x: x + segment_length] for x in range(0, len(tokens), segment_length)]


# Consider now these subsets of tokens as unit to check if the merkmal occurs
# Occurs the merkmal at least once, then it is valid as check
# 'segment_list' should be also iterator?
# After that, count in how many segments (of the partition) the merkmal occurs, i.e. totally

def feature_occurs(segment_list, feature):
    result = [segment for segment in segment_list if feature in segment]
    # int for the number of segments containing the feature
    # 'result' is a list of iterators (lists/tuples, but not sets)
    return len(result), result


# Within the two partitions sort the texts so that
# the text with the highest number of segments containing the feature on top
# and the text with the lowest number of segments containing the feature at the bottom








# Printing the output of the functions
if __name__ == '__main__':
    # this will we deleted when the script is complete
    print('Executing as standalone script')
    print(f'This is the the output of the function: {get_cwd()}')
