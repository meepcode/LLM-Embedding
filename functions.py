import re

# Tokenize
def tokenize(text):
    token_pattern = re.compile(r"\w+|[^\w\s]")
    tokens = re.findall(token_pattern, text)
    return tokens