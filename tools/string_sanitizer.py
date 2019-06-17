import spacy
from spacy.lang.en import English
import re
# from detex import detex

nlp = English()

p_newline = re.compile(r'\n')
def remove_new_line(s):
    return p_newline.sub(' ', s)

p_codetag = re.compile(r'(<code>)(.|\n)*?(<\/code>)')
def remove_code_tags(s):
    return p_codetag.sub(' ', s)

p_htmltag = re.compile(r'<.*?>')
def remove_html_tags(s):
    return p_htmltag.sub(' ', s)

p1 = re.compile(r'\\(begin)(.*?)\\(end)(.*?)(\})')  # Remove \begin ... \end
p2 = re.compile(r'(\$\$)(.*?)(\$\$)')  # Remove $$ ... $$
p3 = re.compile(r'(\$)(.*?)(\$)')  # Remove $ ... $

def remove_latex(s):

    """ Given a string, apply regex to remove Latex """

    s = p1.sub('', s)
    s = p2.sub('', s)
    s = p3.sub('', s)
    return s
    

def sanitize_text(s):

    """ Given long text. 
        Remove newline, tag and latex. 
        Then Tokenize. 

        Args:
        - A Text String

        Return:
        - A clean text
    """

    # Remove Tag and Latex
    s = remove_new_line(s)
    s = remove_code_tags(s)
    s = remove_html_tags(s)
    s = remove_latex(s)

    # Tokenization and Merge it Back
    s = s.lower()
    doc = nlp(s)
    tokens = [ token.text for token in doc if not token.is_space ]

    return " ".join(tokens)


def sanitize_tags(tags):
    
    """ Given a string of tag, 
        split and return a clean string of tags

    """

    tokens = re.split('<|>', tags)
    arr = [ token for token in tokens if len(token) > 0 ]
    return " ".join(arr)