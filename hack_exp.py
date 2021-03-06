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

def get_random_baselines(n_trials=5):
    """Returns
    N random baselines, with its test_results.txt
    """
    if not os.path.exists("random-baseline/"):
            os.makedirs("random-baseline/")
    r = 0.3
    ori_sents = joblib.load('data/train-isw-sentences.pkl')
    ori_labels = joblib.load('data/train-isw-labels.pkl')
    assert len(ori_sents) ==len(ori_labels)

    # Zip the sents and its tags
    train_set = []
    for sent, label in zip(ori_sents, ori_labels):
        train_set.append((sent, label))
    assert len(train_set) == len(ori_sents)

    len_subsample = int(len(train_set)*r)

    # Save random samples
    for i in range(n_trials):
        random_set = choices(train_set, k=len_subsample)
        joblib.dump(random_set, 'random-baseline/random-train-r{}.pkl'.format(i))

    # Train and eval random baseline models
    for i in range(n_trials):
        train_script = "python run_ner.py --output_dir random-baseline/trial{}_model/ --max_seq_length 128 --do_train --do_subtrain --subtrain_dir random-baseline/random-train-r{}.pkl".format(i, i)
        os.system(train_script)

        eval_script = "python run_ner.py --output_dir random-baseline/trial{}_model/ --do_eval --eval_on test --eval_dir random-baseline/eval_monitor/ --it_prefix {}".format(i, i)
        os.system(eval_script)

# Get Co-Train result with fix amount of unlabeled samples setting, u=200,000
def get_random_co_train_result_fix_u(n_trials = 5):
    for i in range(n_trials):
        logger.info(" ***** Selecting ext_data, trial:{}***** ".format(i))
        ext_output_data_dir = "random-co-train/co-ext-data/ext-data-t{}".format(i)
        data_script = "python run_cotrain.py --ext_output_dir {} --modelA_dir small-models --modelB_dir onto_model --de_unlabel_dir machine_translation/2017_de_sents.txt --en_unlabel_dir machine_translation/2017_en_sents.txt --k 1000 --u 200 --top_n 20 --save_preds --save_agree".format(ext_output_data_dir)
        os.system(data_script)

        logger.info(" ***** Start re-train on ext_data, trial:{} ***** ".format(i))
        ext_output_model_dir = "random-co-train/co-ext-models/ext-model-t{}".format(i)
        train_script = "python run_ner.py --max_seq_length 128 --do_train --extend_L --ext_data_dir {} --ext_output_dir {}".format(ext_output_data_dir, ext_output_model_dir)
        os.system(train_script)

        logger.info(" ***** Evaluate new re-train model, trial:{} ***** ".format(i))
        eval_script = "python run_ner.py --output_dir {} --do_eval --eval_on test --eval_dir random-co-train/eval_monitor/ --it_prefix {}".format(ext_output_model_dir, i)
        os.system(eval_script)

# Get Tri-Train result with fix amount of unlabeled samples setting, u=200,000
def get_random_tri_train_result_fix_u(n_trials = 5):
    logger.info(" ***** Pre :Initializing tri candidates models***** ")
    pre_script = "python hack_tri.py --do_prerequisite"
    os.system(pre_script)
    for i in range(n_trials):
        logger.info(" ***** Start tri-training, trial:{}***** ".format(i))
        tri_script = "python hack_tri.py --u 40000 --tcfd_threshold 0.9 --scfd_threshold 0.4 --eval_dir random-tri-train/eval-trial-{}/".format(i)
        os.system(tri_script)

        rm_script = "rm tri-models/s{1..3}_model/1*.*"
        os.system(rm_script)
        rm_script = "rm tri-ext-models/*/*"
        os.system(rm_script)
        rm_script = "rm sub_data/ext*.pkl"
        os.system(rm_script)
        rm_script = "rmdir tri-ext-models/*"
        os.system(rm_script)

# Get Co-Train result with fix amount of selected samples settings, n=[100, 200, 300, 400, 500]
def get_random_co_train_result_fix_n(n_trials = 5, ext_dir="", ext_sent_dir="", ext_label_dir="", selected_n=500):
    sents = joblib.load(ext_sent_dir)
    labels = joblib.load(ext_label_dir)

    # Zip sents and labels
    train_set = []
    for sent, label in zip(sents, labels):
        train_set.append((sent, label))
    assert len(train_set) == len(sents)

    for i in range(n_trials):
        # get random selected samples with desired amount
        random_set = choices(train_set, k=selected_n)
        sentences = [sent for (sent, label) in random_set]
        labels = [label for (sent, label) in random_set]

        # Save as pkl file for further re-training
        joblib.dump(sentences, '{}/{}_ext_L_A_sents.pkl'.format(ext_dir, selected_n))
        joblib.dump(labels, '{}/{}_ext_L_A_labels.pkl'.format(ext_dir, selected_n))

        # Change the config, for indexing
        with open("{}/cotrain_config.json".format(ext_dir), "r") as jsonFile:
            data = json.load(jsonFile)
        data["Prefix"] = selected_n
        with open("{}/cotrain_config.json".format(ext_dir), "w") as jsonFile:
            json.dump(data, jsonFile)

        logger.info(" ***** Start re-train on ext_data, trial:{} ***** ".format(i))
        ext_output_model_dir = "random-co-train/co-ext-models-fix-n-{}/ext-model-t{}".format(selected_n, i)
        train_script = "python run_ner.py --max_seq_length 128 --do_train --extend_L --ext_data_dir {} --ext_output_dir {}".format(ext_dir, ext_output_model_dir)
        os.system(train_script)

        logger.info(" ***** Evaluate new re-train model, trial:{} ***** ".format(i))
        eval_script = "python run_ner.py --output_dir {} --do_eval --eval_on test --eval_dir random-co-train/eval_monitor-fix-n-{}/ --it_prefix {}".format(ext_output_model_dir, selected_n, i)
        os.system(eval_script)

# Get Tri-Train result with fix amount of selected samples settings, n=[100, 200, 300, 400, 500]
def get_random_tri_train_result_fix_n(n_trials = 5, ext_dir="", ori_dir="", selected_n=500):
    ori = joblib.load(ori_dir)
    ext_all = joblib.load(ext_dir)
    amount_ext = len(ext_all)-len(ori)

    ext = ext_all[-amount_ext:]

    for i in range(n_trials):
        random_set = choices(ext, k=selected_n)
        # Append random ext data to ori and Save as pkl file for further re-training
        random_tri_ext = ori + random_set
        random_tri_ext_dir = "sub_data/random-ext-train-trial-{}.pkl".format(i)
        assert len(random_tri_ext) == len(ori) + selected_n
        joblib.dump(random_tri_ext, random_tri_ext_dir)

        # Retrain with fix amount of selected samples
        ext_output_model_dir = "random-tri-train/tri-ext-models-fix-n-{}/ext-model-t{}".format(selected_n, i)
        train_script = "python run_ner.py --output_dir {} --max_seq_length 128 --do_train --do_subtrain --subtrain_dir {}".format(ext_output_model_dir, random_tri_ext_dir)
        os.system(train_script)

        # Evaluate the re-train model
        eval_script = "python run_ner.py --output_dir {} --do_eval --eval_on test --eval_dir random-tri-train/eval_monitor-fix-n-{}/ --it_prefix {}".format(ext_output_model_dir, selected_n, i)
        os.system(eval_script)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--get_random_baselines",
                    action='store_true',
                    help="Whether to train trials baseline model trained on random selected train set")
    parser.add_argument("--get_random_co_train_result_fix_u",
                    action='store_true',
                    help="Whether to train trials co-models with fix amount of unlabeled samples")
    parser.add_argument("--get_random_tri_train_result_fix_u",
                    action='store_true',
                    help="Whether to train trials tri-models with fix amount of unlabeled samples")
    parser.add_argument("--get_random_co_train_result_fix_n",
                    action='store_true',
                    help="Whether to train trials co-models with fix amount of selected samples")
    parser.add_argument("--get_random_tri_train_result_fix_n",
                    action='store_true',
                    help="Whether to train trials tri-models with fix amount of selected samples")
    parser.add_argument("--n_trials",
                        default=5,
                        type=int,
                        help="the number of trials for experiemt")
    parser.add_argument("--selected_n",
                        default=500,
                        type=int,
                        help="the number of n for experiemt")
    args = parser.parse_args()

    if args.get_random_baselines:
        logger.info(" ***** 1. Pre : Getting random baseline ***** ")
        get_random_baselines(n_trials=args.n_trials)

    if args.get_random_co_train_result_fix_u:
        logger.info(" ***** 2. Pre : Getting random co-models with fix u = 200,000 ***** ")
        get_random_co_train_result_fix_u(n_trials=args.n_trials)
    
    if args.get_random_tri_train_result_fix_u:
        logger.info(" ***** 3. Pre : Getting random tri-models with fix u = 200,000 ***** ")
        get_random_tri_train_result_fix_u(n_trials=args.n_trials)

    if args.get_random_co_train_result_fix_n:
        logger.info(" ***** 4. Pre : Getting random co-models with fix n = {} ***** ".format(args.selected_n))
        get_random_co_train_result_fix_n(n_trials=args.n_trials, ext_dir="random-co-train/co-ext-data/ext-data-t1/", ext_sent_dir="random-co-train/co-ext-data/ext-data-t1/1468_ext_L_A_sents.pkl", ext_label_dir="random-co-train/co-ext-data/ext-data-t1/1468_ext_L_A_labels.pkl", selected_n=args.selected_n)

    if args.get_random_tri_train_result_fix_n:
        logger.info(" ***** 5. Pre : Getting random tri-models with fix n = {} ***** ".format(args.selected_n))
        get_random_tri_train_result_fix_n(n_trials=args.n_trials, ext_dir="sub_data/ext-train-isw-s2.pkl", ori_dir="sub_data/train-isw-s2.pkl", selected_n=args.selected_n)


if __name__ == '__main__':
    main()