import spacy
from spacy.lang.en import English
import re

nlp = English()

def remove_new_line(s):
    p = re.compile(r'\n')
    return p.sub(' ', s)

def remove_html_tags(s):
    p = re.compile(r'<.*?>')
    return p.sub(' ', s)

def remove_latex_tags(s):
    p = re.compile(r'\$\\.*?\$')
    return p.sub(' ', s)

def remove_tags(s):
    x = remove_html_tags(s)
    x = remove_new_line(x)
    return remove_latex_tags(x)

def sanitize(s):
    s = remove_tags(s)
    s = s.lower()
    doc = nlp(s)
    tokens = [ token.text for token in doc if not token.is_space ]
    return " ".join(tokens)

def split_tags(tags):
    tokens = re.split('<|>', tags)
    arr = [ token for token in tokens if len(token) > 0 ]
    return " ".join(arr)