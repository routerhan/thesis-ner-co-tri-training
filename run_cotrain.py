import os
import logging
import argparse
from co_training import CoTraining

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # python run_cotrain.py --ext_output_dir ext_data --modelA_dir baseline_model --modelB_dir onto_model --de_unlabel_dir machine_translation/2017_de_sents.txt --en_unlabel_dir machine_translation/2017_en_sents.txt --k 10 --u 10 --top_n 3 --save_preds --save_agree

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--ext_output_dir",
                        default='ext_data/',
                        type=str,
                        required=True,
                        help="The dir that you save the extended L set.")
    parser.add_argument("--modelA_dir",
                        default='baseline_model/',
                        type=str,
                        required=True,
                        help="The dir of pre-trained model that will be used in the cotraining algorithm on the X1 feature set, e.g. German.")
    parser.add_argument("--modelB_dir",
                        default='onto_model/',
                        type=str,
                        required=True,
                        help="The dir of another pre-trained model can be specified to be used on the X2 feature set, e.g. English.")
    parser.add_argument("--de_unlabel_dir",
                        default='machine_translation/2017_de_sents.txt',
                        type=str,
                        required=True,
                        help="The dir of unlabeled sentences in German.")
    parser.add_argument("--en_unlabel_dir",
                        default='machine_translation/2017_en_sents.txt',
                        type=str,
                        required=True,
                        help="The dir of unlabeled sentences in English.")
    parser.add_argument("--save_preds",
                        action='store_true',
                        help="Whether to save the confident predictions.")
    parser.add_argument("--save_agree",
                        action='store_true',
                        help="Whether to save the agree predictions, aka. the predictions that will be added to L set.")
    parser.add_argument("--top_n",
                        default=5,
                        type=int,
                        help="The number of the most confident examples that will be 'labeled' by each classifier during each iteration")
    parser.add_argument("--k",
                        default=30,
                        type=int,
                        help="The number of iterations. The default is 30")
    parser.add_argument("--u",
                        default=75,
                        type=int,
                        help="The size of the pool of unlabeled samples from which the classifier can choose. Default - 75")
    args = parser.parse_args()

    # Initialize co-training class
    if os.path.exists(args.ext_output_dir) and os.listdir(args.ext_output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.ext_output_dir))
    if not os.path.exists(args.ext_output_dir):
        os.makedirs(args.ext_output_dir)

    co_train = CoTraining(modelA_dir=args.modelA_dir, modelB_dir=args.modelB_dir, save_preds=args.save_preds, top_n=args.top_n, k=args.k, u=args.u)
    compare_agree_list = co_train.fit(ext_output_dir=args.ext_output_dir, de_unlabel_dir=args.de_unlabel_dir, en_unlabel_dir=args.en_unlabel_dir, save_agree=args.save_agree, save_preds=args.save_preds)

    logger.info(" ***** Running Co-Training ***** ")
    logger.info(" Model A = {}".format(args.modelA_dir))
    logger.info(" Model B = {}".format(args.modelB_dir))
    logger.info("Top_n: {}, iteration_k: {}, sample_pool_u: {}".format(args.top_n, args.k, args.u))

    logger.info(" ***** Loading Agree Set ***** ")
    logger.info(" Num of agree samples: {}".format(len(compare_agree_list)))
    
if __name__ == '__main__':
    main()
