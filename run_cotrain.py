import os
import logging
import argparse
from co_training import CoTraining

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--unlabel_dir",
                        default='unlabel_sentences/2017_sentences.pkl',
                        type=str,
                        required=True,
                        help="The dir of unlabeled sentences.")
    parser.add_argument("--modelA_dir",
                        default='isw_model/',
                        type=str,
                        required=True,
                        help="The dir of pre-trained model that will be used in the cotraining algorithm on the X1 feature set, e.g. German.")
    parser.add_argument("--modelB_dir",
                        default='onto_model/',
                        type=str,
                        required=True,
                        help="The dir of another pre-trained model can be specified to be used on the X2 feature set, e.g. English.")
    parser.add_argument("--save_tagging",
                        action='store_true',
                        help="Whether to save the predictions.")
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
    co_train = CoTraining(modelA_dir=args.modelA_dir, modelB_dir=args.modelB_dir, save_tagging=args.save_tagging, top_n=args.top_n, k=args.k, u=args.u)
    unlabeled_sentences = co_train.load_unlabeled_set(unlabeled_dir=args.unlabeled_dir)

    logger.info("***** Running Co-Training *****")
    logger.info(" Model A = {}".format(args.modelA_dir))
    logger.info(" Model B = {}".format(args.modelB_dir))
    logger.info("Top_n: {}, iteration_k: {}, num_unlabel_samples_u: {}".format(args.top_n, args.k, args.u))
    logger.info("***** Loading Unlabeled set *****")
    logger.info(" Num of {} samples: {}".format(args.unlabeled_dir ,len(unlabeled_sentences)))
    for index ,sentence in enumerate(unlabeled_sentences[:3]):
        logger.info(" sentence {} : {}".format(index, sentence))
    
