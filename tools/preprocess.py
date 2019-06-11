import pandas as pd
import numpy as np
from parser import parse, post_columns
from sanitizer import sanitize, split_tags
import os
import sys
import argparse

def filter(data):

    """ Filter out some data
    """
    pass

def merge(questions, answers):

    """ Merge between questions and answers
        Returns:
            Array of item with:
            - Question ID
            - Answer ID
            - Question Title
            - Question Description
            - Question Score
            - Answer Description
            - Answer Score
    """

    arr = []
    for i, question in questions.iterrows():
        qid = question['Id']
        title = question['Title']
        tags = question['Tags']
        question_body = question['Body']
        question_score = question['Score']
        answer_count = question['AnswerCount']
        
        if answer_count == 0: continue
        print(answers['ParentId'] == qid)
        break
        for j, answer in answers[answers['ParentId'] == qid]:
            aid = answer['Id']
            answer_body = answer['Body']
            answer_score = answer['Score']

            arr.append( (qid, aid, title, tags, question_body, question_score, answer_body, answer_score) )

    return pd.DataFrame(arr, columns=['QuestionId', 'AnswerId', 'Title', 'Tags', 'QuestionBody', 'QuestionScore', 'AnswerBody', 'AnswerScore'])



def sanitize_qa(name, src_folder, dst_folder):

    """ Process XML File into CSV 
        Tokenization, Sanitization ...
    """

    src_path = os.path.join(os.getcwd(), src_folder)
    src_category = os.path.join(src_path, name)
    src_file = os.path.join(src_category, "Posts.xml")

    questions_headers, answers_headers = post_columns
    questions, answers = parse(src_file)
    question_df = pd.DataFrame(questions, columns=questions_headers)
    answer_df = pd.DataFrame(answers, columns=answers_headers)

    # Sanitize Question Post
    for index, row in question_df.iterrows():
        row['Body'] = sanitize(row['Body'])
        row['Tags'] = split_tags(row['Tags'])
        row['Title'] = sanitize(row['Title'])

    # Sanitize Answer Post
    for index, row in answer_df.iterrows():
        row['Body'] = sanitize(row['Body'])
    
    return question_df, answer_df

def process(name, src_folder, dst_folder):
    questions, answers = sanitize_qa(name, src_folder, dst_folder)
    df = merge(questions, answers)
    return df

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parse XML into CSV')
    parser.add_argument('--name', action="store", dest="name")
    parser.add_argument('--src', action="store", dest="src_folder")
    parser.add_argument('--dst', action="store", dest="dst_folder")
    
    args = parser.parse_args()
    name = args.name
    src_folder = args.src_folder
    dst_folder = args.dst_folder

    # process(name, src_folder, dst_folder)
    question_df, answer_df = sanitize_qa(name, src_folder, dst_folder)

    dst_path = os.path.join(os.getcwd(), dst_folder)
    dst_file = os.path.join(dst_path, name)
    question_df.to_csv(dst_file + '_questions.csv')
    answer_df.to_csv(dst_file + '_answers.csv')
