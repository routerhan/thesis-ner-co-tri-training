import os
import torch
import numpy as np
import argparse
import logging
import json 
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

from seqeval.metrics import classification_report
from tqdm import tqdm, trange

from utils import BertTrainer, get_hyperparameters

from preprocessor import IswPreprocessor, TweetPreprocessor, convert_examples_to_features, InputFeatures
from transformers import BertTokenizer, BertForTokenClassification
from flask import Flask
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

class Ner(BertForTokenClassification):

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,valid_ids=None,attention_mask_label=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask,head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size,max_len,feat_dim,dtype=torch.float32,device='cpu')
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                    if valid_ids[i][j].item() == 1:
                        jj += 1
                        valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            #attention_mask_label = None
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

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
                        default=512,
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

    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

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

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()

    # Base on the data_dir to load preprocessed sentences, labels and tag2idx
    if "isw" in str(args.data_dir):
        pre = IswPreprocessor(filename=args.data_dir)
    elif "headlines" in str(args.data_dir):
        pre = TweetPreprocessor(filename=args.data_dir)
    # Examples
    sentences = pre.sentences
    labels = pre.labels
    label_list = pre.get_labels()

    # Prepare initial tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=False)
    model = BertForTokenClassification.from_pretrained(args.bert_model)
    model.to(device)

    label_map = {i : label for i, label in enumerate(label_list, 1)}
    print('label_map', label_map)
    if args.do_train:
        train_features = convert_examples_to_features(
            all_sentences=sentences, all_labels=labels, 
            label_list=label_list, max_seq_length=args.max_seq_length, tokenizer=tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num train_features = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        num_train_optimization_steps = int(
            len(train_features) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        logger.info("  Num steps = %d", num_train_optimization_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_valid_ids,all_lmask_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        # Set hyperparameters (optimizer, weight decay, learning rate)
        optimizer_grouped_parameters = get_hyperparameters(model, ff=True)
        optimizer = Adam(optimizer_grouped_parameters, lr=args.learning_rate)
        
        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, valid_ids,l_mask = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids,valid_ids,l_mask)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                # loss = model(
                #     input_ids, 
                #     attention_mask=input_mask, 
                #     labels=label_ids)[0]
                # print('loss', loss)
        # input_ids=None,
        # attention_mask=None,
        # token_type_ids=None,
        # position_ids=None,
        # head_mask=None,
        # inputs_embeds=None,
        # labels=None,
        
                # Forward pass
                # outputs = model(
                #     input_ids,
                #     token_type_ids=None,
                #     attention_mask=input_mask,
                #     labels=label_ids,
                # )
                # loss, tr_logits = outputs[:2]


                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # Compute train loss
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    model.zero_grad()
        
        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        label_map = {i : label for i, label in enumerate(label_list,1)}
        print("saved label_map", label_map)
        model_config = {
            "bert_model":args.bert_model,
            "train_data_dir":args.data_dir,
            "train_batch_size":args.train_batch_size,
            "num_train_epochs":args.num_train_epochs,
            "learning_rate":args.learning_rate,
            "adam_epsilon":args.adam_epsilon,
            "max_grad_norm":args.max_grad_norm,
            "max_seq_length":args.max_seq_length,
            "output_dir":args.output_dir,
            "seed":args.seed,
            "gradient_accumulation_steps":args.gradient_accumulation_steps,
            "num_labels":len(label_map),"label_map":label_map
            }
        json.dump(model_config,open(os.path.join(args.output_dir,"model_config.json"),"w"))

    # else load the saved model for evaluation / prediction
    else:
        model = Ner.from_pretrained(args.output_dir)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir)
    
    model.to(device)

    if args.do_eval:
        _, eval_dataloader = bert_trainer.get_train_and_vaild_dataloader()
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
        label_map = {i : label for i, label in enumerate(label_list,1)}
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
        print('before_y_pred', y_pred)
        print('before_y_true', y_true)
        y_pred = [label_map[p_i] for p in y_pred for p_i in p]
        print('y_pred', y_pred)
        y_true = [label_map[l_li] for l in y_true[0] for l_li in l]
        print('y_true', y_true)
        report = classification_report(y_true, y_pred, digits=4)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
             writer.write(report)

if __name__ == '__main__':
    main()