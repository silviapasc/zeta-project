# Check your current work directory
def get_cwd():
    import os
    cwd = os.getcwd()
    return cwd


# Read the text file
def read_text(filename):
    with open(filename, 'rt', encoding='utf-8') as file:
        return file.read()


# Tokenize the text file
def tokenize(text):
    # maybe a regex would be better hier
    punctuation = ',.;:?@«»<>„“…–/·~!"#$%&|=()*¡[]{}\°’`´'
    tokens = text.split()
    for i, token in enumerate(tokens):
        tokens[i] = token.strip(punctuation)
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
