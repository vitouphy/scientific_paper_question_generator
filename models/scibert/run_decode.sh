export PYTHONPATH=`pwd`

# export PROJECT_FOLDER=/home/vitou/workspace/projects/scientific_paper_QG/models/scibert
# export DATA_FOLDER=/home/vitou/workspace/projects/scientific_paper_QG/data/seq2seq_data_bin

export PROJECT_FOLDER=/Users/vitou/Workspaces/AizawaLab/scientific_question_generation/models/scibert
export DATA_FOLDER=/Users/vitou/Workspaces/AizawaLab/playground/scientific_question_generation/Text-Summarizer-Pytorch/data

export MODEL_NAME=lstm_lstm_001
export LOG=${PROJECT_FOLDER}/logs/${MODEL_NAME}
mkdir -p $LOG

# nohup python3 trainings/decode.py \
# --train_data_path=${DATA_FOLDER}/chunked/train/train_*.bin \
# --eval_data_path=${DATA_FOLDER}/chunked/valid/valid_*.bin \
# --decode_data_path=${DATA_FOLDER}/chunked/valid/valid_*.bin \
# --vocab_path=${DATA_FOLDER}/vocabs.txt \
# --logs=$LOG \
# --checkpoint=${LOG}/checkpoint_79 \
# --output_name=checkpoint_79_beam5 \
# --beam_size=5 &> nohup_${MODEL_NAME}_decode5.out &

python3 trainings/seq2seq_decode.py \
--train_data_path=${DATA_FOLDER}/chunked/valid/valid_*.bin \
--eval_data_path=${DATA_FOLDER}/chunked/valid/valid_*.bin \
--decode_data_path=${DATA_FOLDER}/chunked/valid/valid_*.bin \
--vocab_path=${DATA_FOLDER}/vocabs.txt \
--logs=$LOG \
--checkpoint=${LOG}/checkpoint_89 \
--output_name=checkpoint_89_beam1 \
--beam_size=1
