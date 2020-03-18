import os
import torch
import numpy as np
import pandas as pd
import datetime as dt
import argparse
import json 
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


from seqeval.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tqdm import tqdm, trange

from utils import BertTrainer, get_hyperparameters

from preprocessor.preprocessor import *
from transformers import BertTokenizer, BertForTokenClassification


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default='data/test-full-isw-release.tsv',
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--bert_model", default="bert-base-german-cased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")

    parser.add_argument("--output_dir",
                        default='models/',
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--max_seq_length",
                        default=None,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval or not.")

    parser.add_argument("--eval_on",
                        default="valid",
                        help="Whether to run eval on the dev set or test set.")

    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")

    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Base on the data_dir to load preprocessed sentences, labels and tag2idx
    if "isw" in str(args.data_dir):
        pre = IswPreprocessor(filename=args.data_dir)
    elif "headlines" in str(args.data_dir):
        pre = TweetPreprocessor(filename=args.data_dir)
    sentences, labels = pre.get_list_of_sentences_labels()
    tag2idx, idx2tag = pre.get_tag2idx_idx2tag()

    bert_trainer = BertTrainer(bert_model=args.bert_model, 
                        max_len=args.max_seq_length, 
                        batch_size=args.train_batch_size, 
                        full_finetuning=True, 
                        sentences=sentences, 
                        labels=labels, 
                        tag2idx=tag2idx, 
                        idx2tag=idx2tag)

    train_dataloader, eval_dataloader = bert_trainer.get_train_and_vaild_dataloader()
    if args.do_train:
        # Prepare initial tokenizer and model 
        model = bert_trainer.model
        tokenizer = bert_trainer.tokenizer
        model.to(bert_trainer.device)

        # Set hyperparameters (optimizer, weight decay, learning rate)
        optimizer_grouped_parameters = get_hyperparameters(model, ff=True)
        optimizer = Adam(optimizer_grouped_parameters, lr=args.learning_rate)
        print("Initialized optimizer and set hyperparameters.")
        
        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss, tr_accuracy = 0, 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(bert_trainer.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                # Forward pass
                outputs = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )
                loss, tr_logits = outputs[:2]

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.max_grad_norm)

                # Compute train loss
                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

                # Update parameters
                optimizer.step()
                model.zero_grad()
        
        # Save a trained model and the associated configuration
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        label_map = idx2tag
        # label_map = {key:val for key, val in label_map.items() if val != "O"}
        model_config = {"bert_model":args.bert_model,"max_seq_length":args.max_seq_length,"num_labels":len(label_map),"label_map":label_map}
        json.dump(model_config,open(os.path.join(args.output_dir,"model_config.json"),"w"))

    # else load the saved model for evaluation / prediction
    else:
        model = BertForTokenClassification.from_pretrained(args.output_dir)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir)
    
    model.to(bert_trainer.device)

    if args.do_eval:
        if args.eval_on == "valid":
            pass
        elif args.eval_on == "test":
            # TO DO : deal with unseen data
            pass
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        y_true = []
        y_pred = []
        label_map = idx2tag
        # label_map = {key:val for key, val in label_map.items() if val != "O"}

        for batch in eval_dataloader:
            batch = tuple(t.to(bert_trainer.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                outputs = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )
                tmp_eval_loss, logits = outputs[:2]

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            y_pred.extend([list(p) for p in np.argmax(logits, axis=2)])
            y_true.append(label_ids)
        # Evakuate loss, acc, conf. matrix and report on dev set.
        y_pred = [idx2tag[p_i] for p in y_pred for p_i in p]
        y_true = [idx2tag[l_li] for l in y_true[0] for l_li in l]
        report = classification_report(y_true, y_pred, digits=4)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
             writer.write(report)

if __name__ == '__main__':
    main()