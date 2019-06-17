import xml.etree.ElementTree as ET

# questions_headers = [
#     'Id',
#     'PostTypeId',
#     'AcceptedAnswerId',
#     'CreationDate',
#     'Score',
#     "ViewCount",
#     "Body",
#     "OwnerUserId",
#     "LastEditorUserId",
#     "LastEditDate",
#     "LastActivityDate",
#     "Title",
#     "Tags",
#     "AnswerCount",
#     "CommentCount",
#     "FavoriteCount"
# ]

# answers_headers = [
#     'Id', 
#     'PostTypeId', 
#     'ParentId', 
#     'CreationDate', 
#     'Score', 
#     'Body', 
#     'OwnerUserId', 
#     'LastActivityDate', 
#     'CommentCount'
# ]

headers = [
    'QuestionId',
    'AnswerId',
    'Title',
    'Tags',
    'QuestionBody',
    'QuestionScore',
    'AnswerBody',
    'AnswerScore'
]

# post_columns = [questions_headers, answers_headers]

def parseElement(element):
    arr = []
    for column in headers:
        arr.append(element.get(column))
    return arr


def parse(path):

    """Parse XML file
        Args:
            - Path: path of the file
        Return:
            - Parsed Data into an CSV-like array
    """

    data = []
    root = ET.parse(path).getroot()

    for row in root.findall('row'):
        parse_elements = parseElement(row)
        data.append(parse_elements)

    return data
