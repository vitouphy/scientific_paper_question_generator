import spacy
from spacy.lang.en import English
import re
# from detex import detex

nlp = English()

def remove_new_line(s):
    p = re.compile(r'\n')
    return p.sub(' ', s)

def remove_code_tags(s):
    p = re.compile(r'(<code>)(.|\n)*?(<\/code>)')
    return p.sub(' ', s)


def remove_html_tags(s):
    p = re.compile(r'<.*?>')
    return p.sub(' ', s)

p1 = re.compile(r'\\(begin)(.*?)\\(end)(.*?)(\})')  # Remove \begin ... \end
p2 = re.compile(r'(\$\$)(.*?)(\$\$)')  # Remove $$ ... $$
p3 = re.compile(r'(\$)(.*?)(\$)')  # Remove $ ... $

def remove_latex(s):

    """ Given a string, apply regex to remove Latex """

    s = p1.sub('', s)
    s = p2.sub('', s)
    s = p3.sub('', s)
    return s
    

def remove_tags(s):
    x = remove_html_tags(s)
    x = remove_new_line(x)
    return remove_latex_tags(x)

def test(s):
    print(s)
    print ("=====\n")

    s = remove_code_tags(s)
    print(s)
    print ("=====\n")

    s = remove_html_tags(s)
    print (s)
    print ("=====\n")

    s = remove_latex(s)
    print (s)


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