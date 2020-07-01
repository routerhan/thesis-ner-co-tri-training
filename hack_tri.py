import os
import logging

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info(" ***** 1. Pre : sampling with replacement ***** ")
    os.system("python run_tritrain.py --save_subsample --sample_dir sub_data/ --r 0.7 --dataset isw")

    logger.info(" ***** 2. Learn init 3 classifiers from s1, s2 and s3 ***** ")
    # python run_ner.py --output_dir tri-models/s1_model/ --max_seq_length 128 --do_train --do_subtrain --subtrain_dir sub_data/train-isw-s1.pkl
    subset_ls = ["s1", "s2", "s3"]
    for s in subset_ls:
        script = "python run_ner.py --output_dir tri-models/{}_model/ --max_seq_length 128 --do_train --do_subtrain --subtrain_dir sub_data/train-isw-{}.pkl".format(s, s)
        os.system(script)
    
        # python run_ner.py --output_dir tri-models/s1_model/ --max_seq_length 128 --do_eval --eval_on test
        logger.info(" ***** Evaluate init {} classifiers ***** ".format(s))
        script = "python run_ner.py --output_dir tri-models/{}_model/ --max_seq_length 128 --do_eval --eval_on test".format(s)
        os.system(script)


if __name__ == '__main__':
    main()