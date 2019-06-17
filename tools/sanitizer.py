from parser import parse, headers
import string_sanitizer  
import pandas as pd


def clean_tags():
    pass

def clean_html_tags():
    pass

def sanitize(name, src_file, dst_folder):
    data = parse(src_file)
    df = pd.DataFrame(data, columns=headers)
    # print (df.iloc[336]['QuestionBody'])
    string_sanitizer.test(df.iloc[193]['AnswerBody'])


    # Clean Title, Tag, QuestionBoy, AnswerBodyeach row
    # for index, row in df.iterrows():
    #     row['Body'] = sanitize(row['Body'])
    #     row['Tags'] = split_tags(row['Tags'])
    #     row['Title'] = sanitize(row['Title'])

if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description='Parse XML into CSV and Clean Them')
    # parser.add_argument('--name', action="store", dest="name")
    # parser.add_argument('--src', action="store", dest="src_folder")
    # parser.add_argument('--dst', action="store", dest="dst_folder")
    
    # # Getting parameters argument
    # args = parser.parse_args()
    # name = args.name
    # src_folder = args.src_folder
    # dst_folder = args.dst_folder
    # src_folder = os.path.join(os.getcwd(), src_folder)
    # dst_folder = os.path.join(os.getcwd(), dst_folder)

    name = "ai.stackexchange.com"
    src_file = "/Users/vitou/Workspace/scientific_paper_question_generator/analysis_001/data/ai.stackexchange.com.xml"
    dst_folder = "/Users/vitou/Workspace/scientific_paper_question_generator/analysis_001/data/ai.stackexchange.com.csv"

    sanitize(name, src_file, dst_folder)