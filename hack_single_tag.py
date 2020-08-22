import os
import logging
import json
import joblib
import argparse
from random import choices

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def retrain_on_single_tag_co(tag:str, co_sents_dir:str, co_labels_dir:str, ori_train_data="sub_data/train-isw-s3.pkl", fix_len=50):
    co_sents = joblib.load(co_sents_dir)
    co_labels = joblib.load(co_labels_dir)
    ori_train = joblib.load(ori_train_data)

    co_ext = [(s, l) for (s, l) in zip(co_sents, co_labels)]
    tag_co = [(s, l) for (s, l) in co_ext if "B-{}".format(tag) in l][:fix_len]

    single_ext = ori_train + tag_co
    data_dir = 'tmp/{}-co-{}.pkl'.format(tag, fix_len)
    joblib.dump(single_ext, data_dir)
    prefix = "{}_{}_co".format(tag, fix_len)

    logger.info(" ***** Retrain model with ext : {} of {} type with co-train ***** ".format(fix_len, tag))
    script = "python run_ner.py --output_dir single-tag-models/{}_model/ --max_seq_length 128 --do_train --do_subtrain --subtrain_dir {}".format(prefix, data_dir)
    os.system(script)
    logger.info(" ***** Evaluate model with ext : {} of {} type with co-train ***** ".format(fix_len, tag))
    eval_script = "python run_ner.py --output_dir single-tag-models/{}_model/ --do_eval --eval_on test --eval_dir single-tag-models/eval_monitor/ --it_prefix {}".format(prefix, prefix)
    os.system(eval_script)

def retrain_on_single_tag_tri(tag:str, tri_all_dir="sub_data/ext-train-isw-s3.pkl", ori_train_data="sub_data/train-isw-s3.pkl", fix_len=50):
    tri_all = joblib.load(tri_all_dir)
    ori_train = joblib.load(ori_train_data)
    tri_ext = [item for item in tri_all if item not in ori_train]

    tag_tri = [(s, l) for (s, l) in tri_ext if "B-{}".format(tag) in l][:fix_len]

    single_ext = ori_train + tag_tri
    data_dir = 'tmp/{}-tri-{}.pkl'.format(tag, fix_len)
    joblib.dump(single_ext, data_dir)
    prefix = "{}_{}_tri".format(tag, fix_len)

    logger.info(" ***** Retrain model with ext : {} of {} type with tri-train ***** ".format(fix_len, tag))
    script = "python run_ner.py --output_dir single-tag-models/{}_model/ --max_seq_length 128 --do_train --do_subtrain --subtrain_dir {}".format(prefix, data_dir)
    os.system(script)
    logger.info(" ***** Evaluate model with ext : {} of {} type with tri-train ***** ".format(fix_len, tag))
    eval_script = "python run_ner.py --output_dir single-tag-models/{}_model/ --do_eval --eval_on test --eval_dir single-tag-models/eval_monitor/ --it_prefix {}".format(prefix, prefix)
    os.system(eval_script)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_single_co",
                    action='store_true',
                    help="Whether to train single-tag baseline model with co-training ext")
    parser.add_argument("--do_single_tri",
                    action='store_true',
                    help="Whether to train single-tag baseline model with tri-training ext")

    parser.add_argument("--co_sents_dir",
                        default="random-co-train/co-ext-data/ext-data-t0/1482_ext_L_A_sents.pkl",
                        type=str,
                        help="the sent dir of co-training ext data")
    parser.add_argument("--co_labels_dir",
                        default="random-co-train/co-ext-data/ext-data-t0/1482_ext_L_A_labels.pkl",
                        type=str,
                        help="the label dir of co-training ext data")

    parser.add_argument("--tri_all_dir",
                        default="sub_data/ext-train-isw-s3.pkl",
                        type=str,
                        help="the ext_all dir of tri-training approach")

    parser.add_argument("--fix_len",
                        default=50,
                        type=int,
                        help="the number of single tag be introduced")
    parser.add_argument("--tag",
                        default="PER",
                        type=str,
                        help="the tag for single-tag retraining")
    args = parser.parse_args()

    if args.do_single_co:
        logger.info(" ***** 1. Start Co single-tag retrain : len:{}, tag:{} ***** ".format(args.fix_len, args.tag))
        retrain_on_single_tag_co(tag=args.tag, fix_len=args.fix_len, co_sents_dir=args.co_sents_dir, co_labels_dir=args.co_labels_dir, ori_train_data="sub_data/train-isw-s3.pkl")
    if args.do_single_tri:
        logger.info(" ***** 1. Start Tri single-tag retrain : len:{}, tag:{} ***** ".format(args.fix_len, args.tag))
        retrain_on_single_tag_tri(tag=args.tag, fix_len=args.fix_len, tri_all_dir=args.tri_all_dir, ori_train_data="sub_data/train-isw-s3.pkl")

if __name__ == '__main__':
    main()