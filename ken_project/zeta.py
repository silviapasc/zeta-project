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


# Read the text file
def read_text(filename):
    with open(filename, 'rt', encoding='utf-8') as file:
        return file.read()


# Lowercase text
def lowercase(text):
    return text.lower()


# Tokenize the text file
def tokenize(text):
    import re
    # maybe a regex would be better here; uppercase not removed!
    new_text = re.sub(r'[^\w\s]', '', text)
    tokens = new_text.split()
    return tokens


# Count the frequency for each token within the text file
def word_count(tokens):
    frequencies = {}
    for token in tokens:
        if token in frequencies:
            frequencies[token] += 1
        else:
            frequencies[token] = 1
    return frequencies
