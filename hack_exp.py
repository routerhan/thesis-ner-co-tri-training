import os
import logging
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
        rm_script = "rm tri-ext-models/{1..4}_ext_s*_model/*"
        os.system(rm_script)
        rm_script = "rm tri-ext-models/5_ext_s{1..2}_model/*"
        os.system(rm_script)
        rm_script = "rmdir tri-ext-models/"
        os.system(rm_script)


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
    args = parser.parse_args()

    if args.get_random_baselines:
        logger.info(" ***** 1. Pre : Getting random baseline ***** ")
        get_random_baselines(n_trials=5)

    if args.get_random_co_train_result_fix_u:
        logger.info(" ***** 2. Pre : Getting random co-models with fix u = 200,000 ***** ")
        get_random_co_train_result_fix_u(n_trials=5)
    
    if args.get_random_tri_train_result_fix_u:
        logger.info(" ***** 3. Pre : Getting random tri-models with fix u = 200,000 ***** ")
        get_random_tri_train_result_fix_u(n_trials=5)


if __name__ == '__main__':
    main()