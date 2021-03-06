import joblib
import argparse
import json
import logging
import os
import random
import re

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_transformers import (WEIGHTS_NAME, AdamW, BertConfig,
                                  BertForTokenClassification, BertTokenizer,
                                  WarmupLinearSchedule)
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from preprocessor import IswPreprocessor, OntoPreprocessor, convert_examples_to_features, InputFeatures
from utils import split_data
from seqeval.metrics import classification_report

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class Ner(BertForTokenClassification):

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,valid_ids=None,attention_mask_label=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask,head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size,max_len,feat_dim,dtype=torch.float32,device="cuda" if torch.cuda.is_available() else "cpu")
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

def load_train_data(data_dir, ext_data_dir:str, output_dir:str, extend_L=False, extend_L_tri=False):
    if extend_L:
        with open('{}/cotrain_config.json'.format(ext_data_dir)) as f:
            config = json.load(f)
        prefix = config['Prefix']
    # if extend_L_tri:
    #     with open('{}/tri_config.json'.format(ext_data_dir)) as f:
    #         config = json.load(f)
    #     prefix = config['Prefix']

    if "isw" in str(data_dir):
        dataset = "isw"
        logger.info("***** Loading ISW data *****")
        # Only do train/dev/test split on ISW dataset, Do only the very first time...!
        if extend_L:
            with open('{}/model_config.json'.format(output_dir)) as f:
                config = json.load(f)
            label_list = [label for label in config['label_map'].values()]
            num_labels = config['num_labels']
        else:
            label_list, num_labels = split_data(data_dir=data_dir)
        # Load ISW train data
        # sentences = joblib.load('data/train-{}-sentences.pkl'.format(dataset))
        # labels = joblib.load('data/train-{}-labels.pkl'.format(dataset))
        s = joblib.load('small_data/train-isw-s1.pkl')
        sentences = [sent for (sent, label) in s]
        labels = [label for (sent, label) in s]
        
        logger.info("Origin de L size: %d ", len(sentences))
        if extend_L:
            ext_L_A_sents = joblib.load('{}/{}_ext_L_A_sents.pkl'.format(ext_data_dir, prefix))
            ext_L_A_labels = joblib.load('{}/{}_ext_L_A_labels.pkl'.format(ext_data_dir, prefix))
            sentences = sentences + ext_L_A_sents
            labels = labels + ext_L_A_labels
            logger.info("---Co-training---: Ext de L_ size: + {} = {}".format(len(ext_L_A_sents), len(sentences)))
        # if extend_L_tri:
        #     tri_ext_sents = joblib.load('{}/{}_ext_sents.pkl'.format(ext_data_dir, prefix))
        #     tri_ext_labels = joblib.load('{}/{}_ext_labels.pkl'.format(ext_data_dir, prefix))
        #     sentences = sentences + tri_ext_sents
        #     labels = labels + tri_ext_labels
        #     # TODO : 1. ISW + teachable of S1 subeset + teachable
        #     logger.info("---Tri-training---: Ext teachable L_ size: + {} = {}".format(len(tri_ext_sents), len(sentences)))

    elif "onto" in str(data_dir):
        dataset = "onto"
        logger.info("***** Loading OntoNote 5.0 train data *****")
        pre = OntoPreprocessor(filename=data_dir)
        label_list = pre.get_labels()
        num_labels = len(label_list) + 1
        # Load Onto train data
        sentences = pre.sentences
        labels = pre.labels
        logger.info("Origin en L size: %d", len(sentences))
        if extend_L:
            ext_L_B_sents = joblib.load('{}/{}_ext_L_B_sents.pkl'.format(ext_data_dir, prefix))
            ext_L_B_labels = joblib.load('{}/{}_ext_L_B_labels.pkl'.format(ext_data_dir, prefix))
            sentences = sentences + ext_L_B_sents
            labels = labels + ext_L_B_labels
            logger.info("Ext en L_ size: + {} = {}".format(len(ext_L_B_sents), len(sentences)))

    return label_list, num_labels, sentences, labels

def main():
    parser = argparse.ArgumentParser()

    ## Main parameters
    parser.add_argument("--data_dir",
                        default='data/full-isw-release.tsv',
                        type=str,
                        required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default="bert-base-german-cased", type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default='ner',
                        type=str,
                        required=False,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default='baseline_model/',
                        type=str,
                        required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_dir",
                        default="/",
                        help="If specified, the eval result will save into this dir, i.e. used for monitoring the tri-training result.")
    parser.add_argument("--it_prefix",
                        default="",
                        type=str,
                        help="The prefix for monitoring eval results of tri-training, in the format of it-subset, 1_s1")
    parser.add_argument("--it",
                        default=0,
                        type=int,
                        help="the iteration for tri-training")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    # python run_ner.py --data_dir data/full-isw-release.tsv --bert_model bert-base-german-cased --output_dir tri-models/s1_model/ --max_seq_length 128 --do_train --do_subtrain --subtrain_dir sub_data/train-isw-s1.pkl
    parser.add_argument("--do_subtrain",
                        action='store_true',
                        help="Whether to run subtrain on s1, s2 or s3.")
    parser.add_argument("--subtrain_dir",
                        default="sub_data/train-isw-s1.pkl",
                        help="Dir to run sub-training on the s1, s2 or s3 set.")
    parser.add_argument("--extend_L",
                        action='store_true',
                        help="Whether to extend the train set after co-training.")
    parser.add_argument("--extend_L_tri",
                        action='store_true',
                        help="Whether to extend the train set after tri-training.")
    parser.add_argument("--ext_data_dir",
                        default='',
                        type=str,
                        help="The data directory where the extended dataset is saved.")
    parser.add_argument("--ext_output_dir",
                        default='ext_model',
                        type=str,
                        help="The output directory where the extended model is saved.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval or not.")
    parser.add_argument("--eval_on",
                        default="dev",
                        help="Whether to run eval on the dev set or test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
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
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
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
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.extend_L and not args.extend_L_tri:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.extend_L or args.extend_L_tri:
        if os.path.exists(args.ext_output_dir) and os.listdir(args.ext_output_dir) and args.do_train:
            raise ValueError("Ext model output directory ({}) already exists and is not empty.".format(args.ext_output_dir))
        if not os.path.exists(args.ext_output_dir):
            os.makedirs(args.ext_output_dir)

    task_name = args.task_name.lower()

    num_train_optimization_steps = 0
    if args.do_train:
        label_list, num_labels, sentences, labels = load_train_data(data_dir=args.data_dir, extend_L=args.extend_L, ext_data_dir=args.ext_data_dir, output_dir=args.output_dir)

        # check if do subset training : s1, s2, s3...
        if args.do_subtrain:
            s = joblib.load(args.subtrain_dir)
            sentences = [sent for (sent, label) in s]
            labels = [label for (sent, label) in s]

            if args.extend_L_tri:
                with open('{}/{}_tri_config.json'.format(args.ext_data_dir, args.it)) as f:
                    config = json.load(f)
                prefix = config['Prefix']

                logger.info("Origin Student {} L size: {} ".format(args.subtrain_dir,len(sentences)))
                tri_ext_sents = joblib.load('{}/{}_ext_sents.pkl'.format(args.ext_data_dir, prefix))
                tri_ext_labels = joblib.load('{}/{}_ext_labels.pkl'.format(args.ext_data_dir, prefix))
                sentences = sentences + tri_ext_sents
                labels = labels + tri_ext_labels
                assert len(sentences) == len(labels)
                # TODO : 1. ISW + teachable of S1 subeset + teachable
                logger.info("---Tri-training---: Ext teachable L_ size: + {} = {}".format(len(tri_ext_sents), len(sentences)))
                if args.subtrain_dir.find("s1") != -1:
                    prx = "s1"
                elif args.subtrain_dir.find("s2") != -1:
                    prx = "s2"
                else:
                    prx = "s3"
                ext_train_set = [(sent, label) for sent, label in zip(sentences, labels)]
                joblib.dump(ext_train_set, "sub_data/ext-train-isw-{}.pkl".format(prx))
                logger.info("***** Save ext-train-isw.pkl for next iteration : {} *****".format(prx))

        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
        # train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(sentences) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    if args.do_train:
        # Prepare initialized model
        config = BertConfig.from_pretrained(args.bert_model, num_labels=num_labels, finetuning_task=args.task_name)
        model = Ner.from_pretrained(args.bert_model,
                from_tf = False,
                config = config)

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        model.to(device)
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias','LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                            output_device=args.local_rank,
                                                            find_unused_parameters=True)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            all_sentences=sentences, all_labels=labels, 
            label_list=label_list, max_seq_length=args.max_seq_length, tokenizer=tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(sentences))
        logger.info("  Batch size = %d", args.train_batch_size)
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
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

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

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        if args.extend_L or args.extend_L_tri:
            output_dir = args.ext_output_dir
        else:
            output_dir = args.output_dir

        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        label_map = {i : label for i, label in enumerate(label_list,1)}
        model_config = {
            "num_train_examples":len(sentences),
            "bert_model":args.bert_model,
            "do_lower":args.do_lower_case,
            "train_data_dir":args.data_dir,
            "train_batch_size":args.train_batch_size,
            "num_train_epochs":args.num_train_epochs,
            "learning_rate":args.learning_rate,
            "adam_epsilon":args.adam_epsilon,
            "max_grad_norm":args.max_grad_norm,
            "max_seq_length":args.max_seq_length,
            "output_dir":output_dir,
            "seed":args.seed,
            "gradient_accumulation_steps":args.gradient_accumulation_steps,
            "num_labels":len(label_list)+1,"label_map":label_map
            }
        json.dump(model_config,open(os.path.join(output_dir,"model_config.json"),"w"))
        logger.info("***** Success to save model in dir : {} *****".format(output_dir))
    else:
        # Load a trained model and vocabulary that you have fine-tuned
        if args.extend_L or args.extend_L_tri:
            model = Ner.from_pretrained(args.ext_output_dir)
            tokenizer = BertTokenizer.from_pretrained(args.ext_output_dir, do_lower_case=args.do_lower_case)
        else:
            model = Ner.from_pretrained(args.output_dir)
            tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)

    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # If we want to eval the ext model, save the eval result to the ext_model dir..
        if args.extend_L or args.extend_L_tri:
            output_dir = args.ext_output_dir
        else:
            output_dir = args.output_dir
        
        with open('{}/model_config.json'.format(output_dir)) as f:
            config = json.load(f)
        label_list = [label for label in config['label_map'].values()]

        # Base on the data_dir to get the corresponding eval set
        if "isw" in str(args.data_dir):
            dataset = "isw"
            if args.eval_on == "dev":
                eval_sentences = joblib.load('data/dev-{}-sentences.pkl'.format(dataset))
                eval_labels = joblib.load('data/dev-{}-labels.pkl'.format(dataset))
                eval_label_list = label_list
            elif args.eval_on == "test":
                eval_sentences = joblib.load('data/30-test-{}-sentences.pkl'.format(dataset))
                eval_labels = joblib.load('data/30-test-{}-labels.pkl'.format(dataset))
                eval_label_list = label_list
            else:
                raise ValueError("eval on dev or test set only")

        elif "onto" in str(args.data_dir):
            dataset = "onto"
            if args.eval_on == "dev":
                logger.info("***** Loading OntoNote 5.0 dev data *****")
                pre = OntoPreprocessor(filename='../OntoNotes-5.0-NER-BIO/onto.development.ner')
                eval_sentences = pre.sentences
                eval_labels = pre.labels
                eval_label_list = label_list
            elif args.eval_on == "test":
                logger.info("***** Loading OntoNote 5.0 test data *****")
                pre = OntoPreprocessor(filename='../OntoNotes-5.0-NER-BIO/onto.test.ner')
                eval_sentences = pre.sentences
                eval_labels = pre.labels
                eval_label_list = label_list
            else:
                raise ValueError("eval on dev or test set only")

        # Convert into features for testing
        eval_features = convert_examples_to_features(
            all_sentences=eval_sentences, all_labels=eval_labels, 
            label_list=eval_label_list, max_seq_length=args.max_seq_length, tokenizer=tokenizer)
        logger.info("***** Running evaluation: {} *****".format(args.eval_on))
        logger.info("  Num examples = %d", len(eval_sentences))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_valid_ids,all_lmask_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        y_true = []
        y_pred = []
        label_map = {i : label for i, label in enumerate(label_list,1)}
        for input_ids, input_mask, segment_ids, label_ids,valid_ids,l_mask in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            valid_ids = valid_ids.to(device)
            label_ids = label_ids.to(device)
            l_mask = l_mask.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask,valid_ids=valid_ids,attention_mask_label=l_mask)

            logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()

            for i, label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []
                for j,m in enumerate(label):
                    if j == 0:
                        continue
                    elif label_ids[i][j] == len(label_map):
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                        break
                    else:
                        temp_1.append(label_map[label_ids[i][j]])
                        temp_2.append(label_map[logits[i][j]])
        
        report = classification_report(y_true, y_pred, digits=4)
        logger.info("\n%s", report)
        if args.eval_dir != "/":
            if not os.path.exists(args.eval_dir):
                os.makedirs(args.eval_dir)
            output_eval_file = os.path.join(args.eval_dir, "{}_{}_results.txt".format(args.it_prefix ,args.eval_on))
            with open(output_eval_file, "w") as writer:
                logger.info("***** Save the results to {}: {}_{}_results.txt *****".format(args.eval_dir, args.it_prefix, args.eval_on))
                writer.write(report)
        else:
            output_eval_file = os.path.join(output_dir, "{}_results.txt".format(args.eval_on))
            with open(output_eval_file, "w") as writer:
                logger.info("***** Save the results to {}: {}_results.txt *****".format(output_dir, args.eval_on))
                writer.write(report)


if __name__ == "__main__":
    main()
