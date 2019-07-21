export PYTHONPATH=`pwd`

export PROJECT_FOLDER=/Users/vitou/Workspaces/AizawaLab/scientific_question_generation/models/scibert
export DATA_FOLDER=/Users/vitou/Workspaces/AizawaLab/playground/scientific_question_generation/Text-Summarizer-Pytorch/data
export MODEL_NAME=bert_lstm_001
export LOG=${PROJECT_FOLDER}/logs/${MODEL_NAME}
mkdir -p $LOG

python3 trainings/train.py \
--train_data_path=${DATA_FOLDER}/chunked/train/train_*.bin \
--eval_data_path=${DATA_FOLDER}/chunked/valid/valid_*.bin \
--decode_data_path=${DATA_FOLDER}/chunked/test/test_*.bin \
--vocab_path=${DATA_FOLDER}/vocabs.txt \
--logs=$LOG
