import os
import sys
import argparse
import seaborn as sns
import pandas as pd
import numpy as np
from question_counter import count_question
import matplotlib.pyplot as plt
import matplotlib
from column_analysis import get_arr_info

# matplotlib.use('Agg')

def save_pie_chart(wh_count, aux_count, normal, save_dir):
    total = wh_count + aux_count + normal
    sns.set(style="darkgrid")
    plt.figure()
    fig1, ax1 = plt.subplots()

    def absolute_value(val):
        a  = int(np.round(val/100.*total, 0))
        return a

    sns.set()
    sns.set_palette("Blues")

    ax1.pie(
        [ wh_count, aux_count, normal], 
        labels=['wh-question', 'aux-question', 'normal'],
        autopct=absolute_value
    )
    ax1.axis('equal')
    fig1.savefig(os.path.join(save_dir, 'question_type_chart.png'), dpi=300)

def prepare_save_directory(name):
    save_dir = "../figures/"
    save_dir = os.path.join(save_dir, name)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print("Directory " , save_dir ,  " Created ") 
    
    return save_dir

def read_files(name, question_file, answer_file):
    # Import data files
    questions = pd.read_csv('../data/' + name + '_questions.csv')
    answers = pd.read_csv('../data/' + name + '_answers.csv')

    def get_answer_length(row):
        a = row['Body']
        return len(a.split())

    answers.dropna(subset=['Body'], inplace=True)
    answers['AnswerLength'] = answers.apply(get_answer_length, axis=1)

    def get_title_length(row):
        a = row['Title']
        return len(a.split())

    questions['TitleLength'] = questions.apply(get_title_length, axis=1)

    return questions, answers

def get_answers_per_score(answers, save_dir, render=False):
    # Check how many answers per each score
    num_answers = []
    for i in range(13):
        a_score = answers['Score'] == i
        num = a_score.sum()
        num_answers.append(num)

    plt.figure()
    sns.set()
    sns.set_style("white")
    sns.set_palette("hls")
    fig, ax = plt.subplots()
    ax.bar(range(len(num_answers)), num_answers)
    sns.despine()
    ax.set_xlabel('Score')
    ax.set_ylabel('number of answers')
    plt.savefig(os.path.join(save_dir, 'hist_score.png'), dpi=300)
    if render: plt.show()

    print ("Number of Answer at Score: ")
    for i, v in enumerate(num_answers):
        print ("Score {}: {}".format(i, v))

def process(name, question_file, answer_file, dst):   
    save_dir = prepare_save_directory(name)
    questions, answers = read_files(name, question_file, answer_file)

    # Question's Title Analysis
    titles = questions['Title']
    wh_count, aux_count = count_question(titles)
    normal = len(titles) - wh_count - aux_count
    save_pie_chart(wh_count, aux_count, normal, save_dir) 

    print ("WH-Question: {}".format(wh_count))
    print ("AUX-Question: {}".format(aux_count))
    print ("Total : {}".format(len(titles)))

    # Get Answers of Each Score
    get_answers_per_score(answers, save_dir)

    # Answer Length Analysis
    print ("Answer Length Analysis")
    print ("======================\n")
    get_arr_info(answers.AnswerLength, 'answers', save_dir)

    # Question Title Length Analysis
    print ("Question Title Length Analysis")
    print ("======================\n")
    get_arr_info(questions.TitleLength, 'titles', save_dir)
    

if __name__ == "__main__":

    # Get ArgParse
    parser = argparse.ArgumentParser(description='Run analysis')
    parser.add_argument('--name', action="store", dest="name")
    parser.add_argument('--src', action="store", dest="src_folder")
    parser.add_argument('--dst', action="store", dest="dst_folder")
    args = parser.parse_args()

    # Get parameters
    name = args.name
    src = args.src_folder
    dst = args.dst_folder

    question_file = name + "_questions.csv"
    answer_file = name + "_answers.csv"

    process(name, question_file, answer_file, dst)  # Run
    