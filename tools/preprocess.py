import argparse
from sanitizer import sanitize
import pandas as pd
from parser import headers, parse
import numpy as np
import multiprocessing as mp
import os
from string_sanitizer import sanitize_tags
  
def get_vocabs(df):

    """ Parallel Compute the Word Frequency of a DataFrame """

    num_partitions = 10
    num_cores = mp.cpu_count()
    df_split = np.array_split(df, num_partitions)
    pool = mp.Pool(num_cores)
    dict_list = pool.map(get_vocab_dict, df_split)
    pool.close()
    pool.join()

    word_freq = {}
    for word_dict in dict_list:
        for word, count in word_dict.items():
            if word not in word_freq:
                word_freq[word] = 0
            word_freq[word] += count

    return word_freq        

def get_vocab_dict(df):

    """ Count frequency of each word in the dataframe 
        Args:
        - DataFrame df

        Returns:
        - Word Dictionary
    """

    word_dict = {}
    # Not process, count only token in title and in answer body
    for index, row in df.iterrows():
        # Count token in Answer 
        tokens = row['AnswerBody'].split()
        for token in tokens:
            if token not in word_dict:
                word_dict[token] = 0
            word_dict[token] += 1

        # Count token in Title 
        tokens = row['Title'].split()
        for token in tokens:
            if token not in word_dict:
                word_dict[token] = 0
            word_dict[token] += 1

    return word_dict

def get_tags(df):

    """ Parallel Compute the Tag Frequency of a DataFrame """

    num_partitions = 10
    num_cores = mp.cpu_count()
    df_split = np.array_split(df, num_partitions)
    pool = mp.Pool(num_cores)
    dict_list = pool.map(get_tags_dict, df_split)
    pool.close()
    pool.join()

    tag_freq = {}
    for tag_dict in dict_list:
        for tag, count in tag_dict.items():
            if tag not in tag_freq:
                tag_freq[tag] = 0
            tag_freq[tag] += count

    return tag_freq   

def get_tags_dict(df):

    """ Count frequency of each word in the dataframe 
        Args:
        - DataFrame df

        Returns:
        - Word Dictionary
    """

    tag_dict = {}
    for index, row in df.iterrows():
        tags = row['Tags'].split()
        for tag in tags:
            if tag not in tag_dict:
                tag_dict[tag] = 0
            tag_dict[tag] += 1

    return tag_dict

def save_arr_to_file(items, dst_folder, filename):
    dst_file = os.path.join(dst_folder, filename)
    f = open(dst_file, "w")
    for word, freq in items:
        f.write("{} {}\n".format(word, freq))
    f.close()

def save_train_data(df, dst_folder):

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    
    # save the sources 
    dst_file = os.path.join(dst_folder, "sources.txt")
    fs = open(dst_file, 'w')
    for content in df['AnswerBody']:
        fs.write(content)
        fs.write('\n')
    fs.close()

    # save the targets
    dst_file = os.path.join(dst_folder, "targets.txt")
    ft = open(dst_file, 'w')
    for content in df['Title']:
        ft.write(content)
        ft.write('\n')
    ft.close()

def read_train_data(src_folder, filename):
    file = os.path.join(src_folder, filename)
    data = parse(file)
    df = pd.DataFrame(data, columns=headers)
    return df

if __name__ == "__main__":

    # Parse the arguments
    parser = argparse.ArgumentParser(description='Parse XML into CSV and Clean Them')
    parser.add_argument('model')
    parser.add_argument('--src', action="store", dest="src_folder")
    parser.add_argument('--dst', action="store", dest="dst_folder")
    parser.add_argument('--num_words', type=int, default=10000,
                        action="store", dest="num_words")
    parser.add_argument('--num_tags', type=int, default=10000,
                        action="store", dest="num_tags")
    
    # Getting parameters argument
    args = parser.parse_args()
    model = args.model
    src_folder = args.src_folder
    dst_folder = args.dst_folder
    num_words = args.num_words 
    num_tags = args.num_tags 
    print (model)
    print (num_tags)

    # Reading Train, Dev, Test Files
    train_df = read_train_data(src_folder, 'train.xml')
    dev_df = read_train_data(src_folder, 'dev.xml')
    test_df = read_train_data(src_folder, 'test.xml')

    # Sanitize Training Data
    train_df = sanitize(train_df)
    # train_df.dropna(subset=['AnswerBody'], inplace=True)
    train_df = train_df[train_df['AnswerBody'].apply(len) != 0]

    dev_df = sanitize(dev_df)
    # dev_df.dropna(subset=['AnswerBody'], inplace=True)
    dev_df = dev_df[dev_df['AnswerBody'].apply(len) != 0]

    test_df = sanitize(test_df)
    # test_df.dropna(subset=['AnswerBody'], inplace=True)
    test_df = test_df[test_df['AnswerBody'].apply(len) != 0]

    df = pd.concat([train_df, dev_df, test_df])

    word_freq = get_vocabs(df)
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    top_k_words = sorted_words[:num_words]
    save_arr_to_file(sorted_words, dst_folder, "vocabs.txt")
    save_arr_to_file(top_k_words, dst_folder, "vocabs_{}.txt".format(len(top_k_words)))

    tag_freq = get_tags(df)
    sorted_tags = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)
    top_k_tags = sorted_tags[:num_tags]
    save_arr_to_file(sorted_tags, dst_folder, "tags.txt")
    save_arr_to_file(top_k_tags, dst_folder, "tags_{}.txt".format(len(top_k_tags)))

    if model == "seq2seq":

        train_df.to_csv(os.path.join(dst_folder, 'train.csv'), 
                        columns=headers, index=False)
        dev_df.to_csv(os.path.join(dst_folder, 'dev.csv'), 
                        columns=headers, index=False)
        test_df.to_csv(os.path.join(dst_folder, 'test.csv'), 
                        columns=headers, index=False)

        save_train_data(train_df, os.path.join(dst_folder, "train"))
        save_train_data(dev_df, os.path.join(dst_folder, "dev"))
        save_train_data(test_df, os.path.join(dst_folder, "test"))
