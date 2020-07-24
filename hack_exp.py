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
    if not os.path.exists("random_baseline/"):
            os.makedirs("random_baseline/")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--get_random_baselines",
                    action='store_true',
                    help="Whether to train trials baseline model trained on random selected train set")
    args = parser.parse_args()

    if args.get_random_baselines:
        logger.info(" ***** 1. Pre : Getting random baseline ***** ")
        get_random_baselines(n_trials=5)


if __name__ == '__main__':
    main()