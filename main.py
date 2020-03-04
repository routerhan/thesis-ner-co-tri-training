import os

from preprocessor.preprocessor import *
from utils import BertTrainer

if __name__ == '__main__':
    # Constants
    FILE_NAME = 'data/merged_headlines_annos.compact.tsv'
    # FILE_NAME = 'data/full-isw-release.tsv'
    BERT_MODEL = "bert-base-german-cased"
    MAX_LEN = 75
    BATCH_SIZE = 32
    FULL_FINETUNING = True

    # Constants for training
    EPOCHS = 4
    MAX_GRAD_NORM = 1.0
    LR = 3e-5

    # Base on the FILE_NAME to load preprocessed sentences, labels and tag2idx
    if "isw" in str(FILE_NAME):
        pre = IswPreprocessor(filename=FILE_NAME)
    elif "headlines" in str(FILE_NAME):
        pre = TweetPreprocessor(filename=FILE_NAME)
    
    sentences, labels = pre.get_list_of_sentences_labels()
    tag2idx, idx2tag = pre.get_tag2idx_idx2tag()


    # pre = TweetPreprocessor(filename=FILE_NAME)
    # sentences, labels, ners_vals = pre.get_list_of_sentences_labels()
    # tag2idx, idx2tag = pre.get_tag2idx_idx2tag(ners_vals)
    

    # Create directory for storing our model checkpoints
    # if not os.path.exists("/tmp/models"):
    #     os.mkdir("/tmp/models")

    bert_trainer = BertTrainer(bert_model=BERT_MODEL, 
                                max_len=MAX_LEN, 
                                batch_size=BATCH_SIZE, 
                                full_finetuning=FULL_FINETUNING, 
                                sentences=sentences, 
                                labels=labels, 
                                tag2idx=tag2idx, 
                                idx2tag=idx2tag)
    # Start training
    bert_trainer.train_and_save_model(epochs=EPOCHS, max_grad_norm=MAX_GRAD_NORM, learning_rate=LR)



