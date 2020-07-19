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
    parser.add_argument("--do_retrain",
                    action='store_true',
                    help="Whether to re-train the model with new adding ext data")
    
    args = parser.parse_args()

    if args.tune_iteration_k:
        logger.info(" ***** Start Co-training, tuning iteration k ***** ")
        k_ls = [200, 500, 1000, 1200, 1500]
        for k in k_ls:
            logger.info(" ***** Iteration k : {} ***** ".format(k))
            ext_output_data_dir = "ext_data/ext_k_{}".format(k)
            if os.path.exists(ext_output_data_dir) and os.listdir(ext_output_data_dir):
                pass
            else:
                logger.info(" ***** Selecting ext_data ***** ")
                script = "python run_cotrain.py --ext_output_dir {} --modelA_dir small-models --modelB_dir onto_model --de_unlabel_dir machine_translation/2017_de_sents.txt --en_unlabel_dir machine_translation/2017_en_sents.txt --k {} --u 200 --top_n 10 --save_preds --save_agree".format(ext_output_data_dir, k)
                os.system(script)
                logger.info(" ***** Ext data saved in : {} ***** ".format(ext_output_data_dir))

        if args.do_retrain:
            for k in k_ls:
                logger.info(" ***** Start re-train on ext_data ***** ")
                ext_output_model_dir = "co-models/ext_k_{}_model/".format(k)
                ext_output_data_dir = "ext_data/ext_k_{}".format(k)
                script = "python run_ner.py --max_seq_length 128 --do_train --extend_L --ext_data_dir {} --ext_output_dir {}".format(ext_output_data_dir, ext_output_model_dir)
                os.system(script)
                logger.info(" ***** Ext model saved in : {} ***** ".format(ext_output_model_dir))

                logger.info(" ***** Evaluate new re-train model ***** ")
                script = "python run_ner.py --max_seq_length 128 --do_eval --eval_on test --extend_L --ext_output_dir {}".format(ext_output_model_dir)
                os.system(script)

    if args.tune_top_n:
        logger.info(" ***** Start Co-training, tuning top_n ***** ")
        t_ls = [5, 10, 15, 20, 25]
        for t in t_ls:
            logger.info(" ***** Top_N: {} ***** ".format(t))
            ext_output_data_dir = "ext_data/ext_top_{}".format(t)
            if os.path.exists(ext_output_data_dir) and os.listdir(ext_output_data_dir):
                pass
            else:
                logger.info(" ***** Selecting ext_data ***** ")
                script = "python run_cotrain.py --ext_output_dir {} --modelA_dir small-models --modelB_dir onto_model --de_unlabel_dir machine_translation/2017_de_sents.txt --en_unlabel_dir machine_translation/2017_en_sents.txt --k 1000 --u 200 --top_n {} --save_preds --save_agree".format(ext_output_data_dir, t)
                os.system(script)
                logger.info(" ***** Ext data saved in : {} ***** ".format(ext_output_data_dir))

        if args.do_retrain:
            for t in t_ls:
                logger.info(" ***** Start re-train on ext_data ***** ")
                ext_output_model_dir = "co-models/ext_top_{}_model/".format(t)
                ext_output_data_dir = "ext_data/ext_top_{}".format(t)
                script = "python run_ner.py --max_seq_length 128 --do_train --extend_L --ext_data_dir {} --ext_output_dir {}".format(ext_output_data_dir, ext_output_model_dir)
                os.system(script)
                logger.info(" ***** Ext model saved in : {} ***** ".format(ext_output_model_dir))

                logger.info(" ***** Evaluate new re-train model ***** ")
                script = "python run_ner.py --max_seq_length 128 --do_eval --eval_on test --extend_L --ext_output_dir {}".format(ext_output_model_dir)
                os.system(script)

    if args.tune_pool_value:
        logger.info(" ***** Start Co-training, tuning tune_pool_value u ***** ")
        u_ls = [50, 100, 150, 200, 250]
        for u in u_ls:
            logger.info(" ***** Pool_u: {} ***** ".format(u))
            ext_output_data_dir = "ext_data/ext_u_{}".format(u)
            if os.path.exists(ext_output_data_dir) and os.listdir(ext_output_data_dir):
                pass
            else:
                logger.info(" ***** Selecting ext_data ***** ")
                script = "python run_cotrain.py --ext_output_dir {} --modelA_dir small-models --modelB_dir onto_model --de_unlabel_dir machine_translation/2017_de_sents.txt --en_unlabel_dir machine_translation/2017_en_sents.txt --k 1000 --u {} --top_n 10 --save_preds --save_agree".format(ext_output_data_dir, u)
                os.system(script)
                logger.info(" ***** Ext data saved in : {} ***** ".format(ext_output_data_dir))

        if args.do_retrain:
            for u in u_ls:
                logger.info(" ***** Start re-train on ext_data ***** ")
                ext_output_model_dir = "co-models/ext_u_{}_model/".format(u)
                ext_output_data_dir = "ext_data/ext_u_{}".format(u)
                script = "python run_ner.py --max_seq_length 128 --do_train --extend_L --ext_data_dir {} --ext_output_dir {}".format(ext_output_data_dir, ext_output_model_dir)
                os.system(script)
                logger.info(" ***** Ext model saved in : {} ***** ".format(ext_output_model_dir))

                logger.info(" ***** Evaluate new re-train model ***** ")
                script = "python run_ner.py --max_seq_length 128 --do_eval --eval_on test --extend_L --ext_output_dir {}".format(ext_output_model_dir)
                os.system(script)

if __name__ == '__main__':
    main()
