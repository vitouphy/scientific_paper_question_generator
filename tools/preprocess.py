import argparse
from sanitizer import sanitize
import pandas as pd
from parser import headers
import numpy as np
import multiprocessing as mp
import os
  
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

def choose_n_words(word_freq, n):
    sort_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    top_n_words = sort_words[:n]
    word_list = [ word for (word, count) in top_n_words ]
    return sorted(word_list)

def save_word_to_file(words, dst_folder):
    dst_file = os.path.join(dst_folder, 'vocab.txt')
    f = open(dst_file, "w")
    for word in words:
        f.write(word)
        f.write('\n')
    f.close()

def split_and_save_df_to_files(df, data_ratio, dst_folder):

    # Fill in data if no info provided
    if data_ratio is None:
        total = len(df)
        num_train = int(total * .7)
        num_dev = int(total * .2)
        num_test = total - num_train - num_dev
    else:
        num_train, num_dev, num_test = data_ratio
    
    # Shuffling the data
    df = df.sample(frac=1).reset_index(drop=True) 

    # Split up data into 3 partitions   
    df_train = df[:num_train]
    df_dev = df[num_train:num_train + num_dev]
    df_test = df[num_train + num_dev :]

    # Use 3 Process to Save Data to 3 Diff Files
    num_cores = mp.cpu_count()
    pool = mp.Pool(num_cores)
    rows = [
        (df_train, os.path.join(dst_folder, "train")),
        (df_dev, os.path.join(dst_folder, "dev")),
        (df_test, os.path.join(dst_folder, "test"))
    ]
    dict_list = [pool.apply(save_train_data, args=(df, path)) for (df, path) in rows]
    pool.close()
    pool.join()

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parse XML into CSV and Clean Them')
    parser.add_argument('--mode', action="store", dest="mode", 
                        help='type of mode (sanitize or full)')
    parser.add_argument('--split_ratio', action="store", dest="split_ratio",
                        help='split into train/dev/test in int number')
    parser.add_argument('--name', action="store", dest="name")
    parser.add_argument('--src', action="store", dest="src_file")
    parser.add_argument('--dst', action="store", dest="dst_folder")

    # Case for mode == full preprocessing
    parser.add_argument('--num_words', type=int, default=10000,
                        action="store", dest="num_words")

    # Getting parameters argument
    args = parser.parse_args()
    name = args.name
    src_file = args.src_file
    dst_folder = args.dst_folder
    num_words = args.num_words
    mode = args.mode
    print(num_words)

    data_ratio = None
    if args.split_ratio is not None:
        data_ratio = args.split_ratio.split('/')

    if mode == "sanitize":
        dst_file = os.path.join(dst_folder, name + ".csv")
        df = sanitize(name, src_file, dst_file)
        df.to_csv(dst_file, columns=headers, index=False)

    elif mode == "full":
        dst_file = os.path.join(dst_folder, name + ".csv")
        df = sanitize(name, src_file, dst_file)
        df.to_csv(dst_file, columns=headers, index=False)

        # Prepare the vocab file
        word_freq = get_vocabs(df)
        words = choose_n_words(word_freq, num_words)
        save_word_to_file(words, dst_folder)

        # Prepare the training/dev/test file
        split_and_save_df_to_files(df, data_ratio, dst_folder)

