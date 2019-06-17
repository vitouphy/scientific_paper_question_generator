from parser import parse, headers
from string_sanitizer import sanitize_text, sanitize_tags
import pandas as pd
import time
import multiprocessing as mp
import argparse
import os

def sanitize_row(item):
    idx, row = item
    row['Title'] = sanitize_text(row['Title'])
    row['QuestionBody'] = sanitize_text(row['QuestionBody'])
    row['AnswerBody'] = sanitize_text(row['AnswerBody'])
    row['Tags'] = sanitize_tags(row['Tags'])
    return row

def sanitize(name, src_file, dst_file):

    data = parse(src_file)
    df = pd.DataFrame(data, columns=headers)
    
    startTime = time.time()
    pool = mp.Pool(mp.cpu_count())

    # Clean Title, Tag, QuestionBoy, AnswerBodyeach row
    # for index, row in df.iterrows():
    #     row['Title'] = sanitize_text(row['Title'])
    #     row['QuestionBody'] = sanitize_text(row['QuestionBody'])
    #     row['AnswerBody'] = sanitize_text(row['AnswerBody'])
    #     row['Tags'] = sanitize_tags(row['Tags'])

    result = pool.map(sanitize_row, df.iterrows())
    result = pd.DataFrame(result, columns=headers)
    pool.close()
    pool.join()

    endTime = time.time()
    print("total conversion time: {:0.2f} s".format(endTime - startTime))

    result.to_csv(dst_file, columns=headers, index=False)
    # df.to_csv(dst_file, columns=headers, index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parse XML into CSV and Clean Them')
    parser.add_argument('--name', action="store", dest="name")
    parser.add_argument('--src', action="store", dest="src_folder")
    parser.add_argument('--dst', action="store", dest="dst_folder")
    
    # Getting parameters argument
    args = parser.parse_args()
    name = args.name
    src_folder = args.src_folder
    src_folder = os.path.join(os.getcwd(), src_folder)
    src_file = os.path.join(src_folder, name + ".xml")

    dst_folder = args.dst_folder
    dst_folder = os.path.join(os.getcwd(), dst_folder)
    dst_file = os.path.join(dst_folder, name + ".csv")

    # name = "ai.stackexchange.com"
    # src_file = "/Users/vitou/Workspace/scientific_paper_question_generator/analysis_001/data/ai.stackexchange.com.xml"
    # dst_folder = "/Users/vitou/Workspace/scientific_paper_question_generator/analysis_001/data/ai.stackexchange.com.csv"

    sanitize(name, src_file, dst_file)