import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
from question_counter import count_question
from column_analysis import get_arr_info

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

def export_vocab(questions, answers):

    print ("Vocab Size: ")
    print ("==========")
    
    vocab = {}
    # Get token from questions title
    for titles in questions['Title']:
        for token in titles.split():
            if token not in vocab:
                vocab[token] = 0
            vocab[token] += 1     
    print ("num of vocabs in question: {}".format(len(vocab)))

    # Get token from questions title
    for body in answers['Body']:
        for token in body.split():
            if token not in vocab:
                vocab[token] = 0
            vocab[token] += 1    
    print ("num of vocabs: {}".format(len(vocab)))
    
    return vocab

def create_wordcloud(dict, save_dir):
    # Create and generate a word cloud image:
    wordcloud = WordCloud(width=1200, height=700, max_font_size=100, max_words=100, background_color="white")
    wordcloud = wordcloud.generate_from_frequencies(dict)

    # Display the generated image:
    plt.figure(figsize=(12,7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, 'wordcloud.png'), dpi=600)

def save_pie_chart(wh_count, yes_no_count, normal, save_dir):
    total = wh_count + yes_no_count + normal
    sns.set(style="darkgrid")
    plt.figure()
    fig1, ax1 = plt.subplots()

    def absolute_value(val):
        a  = int(np.round(val/100.*total, 0))
        return a

    sns.set()
    sns.set_palette("Blues")

    ax1.pie(
        [ wh_count, yes_no_count, normal], 
        labels=['wh-question', 'yes-no-question', 'normal'],
        autopct=absolute_value
    )
    ax1.axis('equal')
    fig1.savefig(os.path.join(save_dir, 'question_type_chart.png'), dpi=300)

def prepare_save_output_directory(name):
    save_dir = "../outputs/"
    save_dir = os.path.join(save_dir, name)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print("Directory " , save_dir ,  " Created ") 
    
    return save_dir

# def prepare_save_directory(name):
#     save_dir = "../figures/"
#     save_dir = os.path.join(save_dir, name)

#     if not os.path.exists(save_dir):
#         os.mkdir(save_dir)
#         print("Directory " , save_dir ,  " Created ") 
    
#     return save_dir

def read_files(train_file, dev_file, test_file):
    # Import data files
    df_train = pd.read_csv(train_file)
    df_dev = pd.read_csv(dev_file)
    df_test = pd.read_csv(test_file)
    df = pd.concat([df_train, df_dev, df_test])

    def get_answer_length(row):
        a = row['AnswerBody']
        return len(a.split())
    df['AnswerLength'] = df.apply(get_answer_length, axis=1)

    def get_title_length(row):
        a = row['Title']
        return len(a.split())
    df['TitleLength'] = df.apply(get_title_length, axis=1)

    return df

def get_answers_per_score(df, save_dir, render=False):
    # Check how many answers per each score
    num_answers = []
    for i in range(13):
        a_score = df['AnswerScore'] == i
        num = a_score.sum()
        num_answers.append(num)

    sns.set()
    sns.set_style("white")
    sns.set_palette("hls")
    fig, ax = plt.subplots()
    ax.bar(range(len(num_answers)), num_answers)
    sns.despine()
    ax.set_xlabel('Score')
    ax.set_ylabel('number of answers')
    plt.savefig(os.path.join(save_dir, 'hist_score.png'), dpi=300)


    # print ("Number of Answer at Score: ")
    # for i, v in enumerate(num_answers):
    #     print ("Score {}: {}".format(i, v))

def process(train_file, dev_file, test_file, tags_file, dst_folder):   
    # save_dir = prepare_save_directory(dst_folder)
    df = read_files(train_file, dev_file, test_file)

    # Question's Title Analysis
    titles = df['Title']
    wh_count, yes_no_count = count_question(titles)
    total = len(titles)
    normal = total - wh_count - yes_no_count
    save_pie_chart(wh_count, yes_no_count, normal, dst_folder)

    print ("WH-Question: {}".format(wh_count))
    print ("Yes-No-Question: {}".format(yes_no_count))
    print ("Total : {}".format(len(titles)))

    # Get Answers of Each Score
    get_answers_per_score(df, dst_folder)

    # Answer Length Analysis
    print ("\n")
    print ("======================")
    print ("Answer Length Analysis")
    get_arr_info(df.AnswerLength, 'answers', dst_folder)

    # Question Title Length Analysis
    print ("\n")
    print ("======================")
    print ("Question Title Length Analysis")
    get_arr_info(df.TitleLength, 'titles', dst_folder)

    # Analysis the tags
    tags_dict = {}
    f = open(tags_file, 'r')
    for line in f.readlines():
        elements = line.split()
        word = elements[0]
        freq = int(elements[1])
        tags_dict[word] = freq

    create_wordcloud(tags_dict, dst_folder) 
    
    # tags_dict = {}
    # for tags in questions['Tags']:
    #     for t in tags.split():
    #         if t not in tags_dict:
    #             tags_dict[t] = 0
    #         tags_dict[t] += 1

    # create_wordcloud(tags_dict, save_dir) 
    # vocab = export_vocab(questions, answers)

    # dict = {
    #     'vocab': vocab, 
    #     'tags': tags_dict
    # }
    # out_dir = prepare_save_output_directory(name)
    # out_file = os.path.join(out_dir, "analysis.pickle")
    # pickle_out = open(out_file, "wb")
    # pickle.dump(dict, pickle_out)
    # pickle_out.close()
    

if __name__ == "__main__":

    # Get ArgParse
    parser = argparse.ArgumentParser(description='Run analysis')
    parser.add_argument('--src_train', action="store", dest="train_file")
    parser.add_argument('--src_dev', action="store", dest="dev_file")
    parser.add_argument('--src_test', action="store", dest="test_file")
    parser.add_argument('--src_tags', action="store", dest="tags_file")
    parser.add_argument('--dst', action="store", dest="dst_folder")
    args = parser.parse_args()

    # Get parameters
    train_file = args.train_file
    dev_file = args.dev_file
    test_file = args.test_file
    tags_file = args.tags_file
    dst_folder = args.dst_folder

    process(train_file, dev_file, test_file, tags_file, dst_folder)  # Run
    
