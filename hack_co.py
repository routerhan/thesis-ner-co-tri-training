import os
import logging
import argparse

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune_iteration_k",
                    action='store_true',
                    help="Whether to tune number of iterations k value")
    parser.add_argument("--tune_top_n",
                    action='store_true',
                    help="Whether to tune number of top_n")
    parser.add_argument("--tune_pool_value",
                    action='store_true',
                    help="Whether to tune the pool value u")
    
    args = parser.parse_args()

    if args.tune_iteration_k:
        logger.info(" ***** Start Co-training, tuning iteration k ***** ")
        k_ls = [500, 1000, 1500, 2000, 2500]
        for k in k_ls:
            logger.info(" ***** Iteration k : {} ***** ".format(k))
            ext_output_dir = "ext_data/ext_k_{}".format(k)
            script = "python run_cotrain.py --ext_output_dir {} --modelA_dir baseline_model --modelB_dir onto_model --de_unlabel_dir machine_translation/2017_de_sents.txt --en_unlabel_dir machine_translation/2017_en_sents.txt --k {} --u 200 --top_n 10 --save_preds --save_agree".format(ext_output_dir, k)
            os.system(script)
            logger.info(" ***** Ext model saved in : {} ***** ".format(ext_output_dir))

if __name__ == '__main__':
    main()