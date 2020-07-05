import os
import logging
import argparse

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_prerequisite",
                    action='store_true',
                    help="Whether to do all the set up, i.e. sampling and get candidates clfs.")
    
    args = parser.parse_args()

    if args.do_prerequisite:
        logger.info(" ***** 1. Pre : sampling with replacement ***** ")
        os.system("python run_tritrain.py --save_subsample --sample_dir sub_data/ --r 0.7 --dataset isw")

        logger.info(" ***** 2. Pre : Learn init 3 classifiers from s1, s2 and s3 ***** ")
        # python run_ner.py --output_dir tri-models/s1_model/ --max_seq_length 128 --do_train --do_subtrain --subtrain_dir sub_data/train-isw-s1.pkl
        subset_ls = ["s1", "s2", "s3"]
        for s in subset_ls:
            script = "python run_ner.py --output_dir tri-models/{}_model/ --max_seq_length 128 --do_train --do_subtrain --subtrain_dir sub_data/train-isw-{}.pkl".format(s, s)
            os.system(script)
        
            # python run_ner.py --output_dir tri-models/s1_model/ --max_seq_length 128 --do_eval --eval_on test
            logger.info(" ***** 3. Pre: Evaluate init {} classifiers ***** ".format(s))
            script = "python run_ner.py --output_dir tri-models/{}_model/ --max_seq_length 128 --do_eval --eval_on test".format(s)
            os.system(script)
    else:
        logger.info(" ***** Start Tri-training pipeline ***** ")
        script = "python run_tritrain.py --U machine_translation/2017_de_sents.txt --u 200 --mi_dir tri-models/s1_model/ --mj_dir tri-models/s2_model/ --mk_dir tri-models/s3_model/ --tcfd_threshold 0.7 --scfd_threshold 0.6 --r_t 0.1 --r_s 0.1"
        os.system(script)


if __name__ == '__main__':
    main()