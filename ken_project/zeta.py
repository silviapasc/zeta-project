# Check your current working directory
def get_cwd():
    import os
    cwd = os.getcwd()
    return cwd


# Set required working directory
def set_cwd(path):
    import os
    new_path = path
    return os.chdir(new_path)


# Read the text file -> str
def read_text(filename):
    with open(filename, 'rt', encoding='utf-8') as file:
        return file.read()


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


# Printing the output of the functions
if __name__ == '__main__':
    # this will we deleted when the script is complete
    print('Executing as standalone script')
    print(f'This is the the output of the function: {get_cwd()}')
