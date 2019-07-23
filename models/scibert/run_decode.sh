export PYTHONPATH=`pwd`

# export PROJECT_FOLDER=/home/vitou/workspace/projects/scientific_paper_QG/models/scibert
# export DATA_FOLDER=/home/vitou/workspace/projects/scientific_paper_QG/data/seq2seq_data_bin

export PROJECT_FOLDER=/Users/vitou/Workspaces/AizawaLab/scientific_question_generation/models/scibert
export DATA_FOLDER=/Users/vitou/Workspaces/AizawaLab/playground/scientific_question_generation/Text-Summarizer-Pytorch/data

export MODEL_NAME=bert_lstm_003
export LOG=${PROJECT_FOLDER}/logs/${MODEL_NAME}
mkdir -p $LOG

python3 trainings/decode.py \
--train_data_path=${DATA_FOLDER}/chunked/train/train_*.bin \
--eval_data_path=${DATA_FOLDER}/chunked/valid/valid_*.bin \
--decode_data_path=${DATA_FOLDER}/chunked/test/test_*.bin \
--vocab_path=${DATA_FOLDER}/vocabs.txt \
--logs=$LOG \
--checkpoint=${LOG}/checkpoint_2179 \
--output_name=checkpoint_2179_beam5 \
--beam_size=10
