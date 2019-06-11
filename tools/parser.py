import xml.etree.ElementTree as ET

questions_headers = [
    'Id',
    'PostTypeId',
    'AcceptedAnswerId',
    'CreationDate',
    'Score',
    "ViewCount",
    "Body",
    "OwnerUserId",
    "LastEditorUserId",
    "LastEditDate",
    "LastActivityDate",
    "Title",
    "Tags",
    "AnswerCount",
    "CommentCount",
    "FavoriteCount"
]

answers_headers = [
    'Id', 
    'PostTypeId', 
    'ParentId', 
    'CreationDate', 
    'Score', 
    'Body', 
    'OwnerUserId', 
    'LastActivityDate', 
    'CommentCount'
]

post_columns = [questions_headers, answers_headers]

def parseElement(element, type):
    arr = []
    if type == 1:
        keys = questions_headers
    else:
        keys = answers_headers
    for key in keys:
        arr.append(element.get(key))
    return arr


def parse(path):

    """Parse XML file
        Args:
            - Path: path of the file
        Return:
            - Parsed Questions
            - Parsed Answers
    """

    questions = []
    answers = []
    root = ET.parse(path).getroot()

    for row in root.findall('row'):
        type = int(row.get('PostTypeId'))
        parse_elements = parseElement(row, type)
        if type == 1:  # Question Type
            questions.append(parse_elements)
        else:
            answers.append(parse_elements)
    
    return questions, answers
