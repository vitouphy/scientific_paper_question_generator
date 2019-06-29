import os
import shutil
import collections
import tqdm
from tensorflow.core.example import example_pb2
import struct
import random
import shutil
import argparse

src_folder = "unfinished"
finished_path = "finished"
chunk_path = "chunked"

vocab_path = "data/vocabs.txt"
VOCAB_SIZE = 100000000

CHUNK_SIZE = 15000 # num examples per chunk, for the chunked data
train_bin_path = ""
valid_bin_path = ""
test_bin_path = ""

def make_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

def write_to_bin(article_path, abstract_path, out_file, vocab_counter = None):

    with open(out_file, 'wb') as writer:

        article_itr = open(article_path, 'r')
        abstract_itr = open(abstract_path, 'r')
        for article in tqdm.tqdm(article_itr):
            article = article.strip()
            abstract = next(abstract_itr).strip()

            tf_example = example_pb2.Example()
            tf_example.features.feature['article'].bytes_list.value.extend([article.encode('utf8')])
            tf_example.features.feature['abstract'].bytes_list.value.extend([abstract.encode('utf8')])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))

            if vocab_counter is not None:
                art_tokens = article.split(' ')
                abs_tokens = abstract.split(' ')
                # abs_tokens = [t for t in abs_tokens if
                #               t not in [SENTENCE_START, SENTENCE_END]]  # remove these tags from vocab
                tokens = art_tokens + abs_tokens
                tokens = [t.strip() for t in tokens]  # strip
                tokens = [t for t in tokens if t != ""]  # remove empty
                vocab_counter.update(tokens)

    if vocab_counter is not None:
        with open(vocab_path, 'w') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')

def creating_finished_data():
    make_folder(finished_path)

    vocab_counter = collections.Counter()

    write_to_bin( os.path.join(src_folder, "train.sources.txt"), 
                  os.path.join(src_folder, "train.targets.txt"), 
                  train_bin_path, vocab_counter  )
    write_to_bin( os.path.join(src_folder, "valid.sources.txt"), 
                  os.path.join(src_folder, "valid.targets.txt"), 
                  valid_bin_path, vocab_counter  )
    write_to_bin( os.path.join(src_folder, "test.sources.txt"), 
                  os.path.join(src_folder, "test.targets.txt"), 
                  test_bin_path, vocab_counter  )

def chunk_file(set_name, chunks_dir, bin_file):
    make_folder(chunks_dir)
    reader = open(bin_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join(chunks_dir, '%s_%04d.bin' % (set_name, chunk)) # new chunk
        with open(chunk_fname, 'wb') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Convert seq2seq data to chunk bins")
    parser.add_argument("--data_folder",
                        dest="data_folder", 
                        required=True,
                        help="Folder for storing raw and clean data")
    args = parser.parse_args()
    data_folder = args.data_folder

    src_folder = os.path.join(data_folder, src_folder)
    finished_path = os.path.join(data_folder, finished_path)
    chunk_path = os.path.join(data_folder, chunk_path)

    train_bin_path = os.path.join(finished_path, "train.bin")
    valid_bin_path = os.path.join(finished_path, "valid.bin")
    test_bin_path  = os.path.join(finished_path, "test.bin")

    # Create bin file for train, val, test
    delete_folder(finished_path)
    creating_finished_data()        #create bin files
    print("Completed creating bin file for train, valid, & test")

    delete_folder(chunk_path)
    chunk_file("train", os.path.join(chunk_path, "train"), train_bin_path)
    chunk_file("valid", os.path.join(chunk_path, "valid"), valid_bin_path)
    chunk_file("test", os.path.join(chunk_path, "test"), test_bin_path)
    print("Completed chunking main bin files into smaller ones")

    #Performing rouge evaluation on 1.9 lakh sentences takes lot of time. So, create mini validation set & test set by borrowing 15k samples each from these 1.9 lakh sentences
    # make_folder(os.path.join(chunk_path, "valid"))
    # make_folder(os.path.join(chunk_path, "test"))
    # bin_chunks = os.listdir(os.path.join(chunk_path, "main_valid"))
    # bin_chunks.sort()
    # samples = random.sample(set(bin_chunks[:-1]), 2)      #Exclude last bin file; contains only 9k sentences
    # valid_chunk, test_chunk = samples[0], samples[1]
    # shutil.copyfile(os.path.join(chunk_path, "main_valid", valid_chunk), os.path.join(chunk_path, "valid", "valid_00.bin"))
    # shutil.copyfile(os.path.join(chunk_path, "main_valid", test_chunk), os.path.join(chunk_path, "test", "test_00.bin"))

    # delete_folder(finished)
    # delete_folder(os.path.join(chunk_path, "main_valid"))




