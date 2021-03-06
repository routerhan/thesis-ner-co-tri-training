import os
import json
import logging
import argparse
import utils
import joblib
from tri_train import TriTraining


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # python run_tritrain.py --save_subsample --sample_dir sub_data/ --r1 0.4 --r2 0.4 --r3 0.2 --dataset isw
    # When you already have models trained on subset of original train set, you can do the following command:
    # python run_tritrain.py --ext_data_dir tri_ext_data --val_on test --U data/dev-isw-sentences.pkl --u 100 --mi_dir tri-models/s1_model/ --mj_dir tri-models/s2_model/ --mk_dir tri-models/s3_model/ --tcfd_threshold 0.7 --scfd_threshold 0.6 --r_t 0.1 --r_s 0.1
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--U",
                        default='data/dev-isw-sentences.pkl',
                        type=str,
                        help="The file of unlabeled set with the format of list of sentences.")
    parser.add_argument("--u",
                        default=100,
                        type=int,
                        help="The pool value of U, to limit the amount of unlabeled samples to use.")
    parser.add_argument("--cos_score_threshold",
                        default=0.9,
                        type=float,
                        help="The similarity socre threshold to check identicality of sequance labeling task.")
    parser.add_argument("--mi_dir",
                        default='tri-models/s1_model/',
                        type=str,
                        help="The model dir trained on subset s1.")
    parser.add_argument("--mj_dir",
                        default='tri-models/s2_model/',
                        type=str,
                        help="The model dir trained on subset s2.")
    parser.add_argument("--mk_dir",
                        default='tri-models/s3_model/',
                        type=str,
                        help="The model dir trained on subset s3.")
    parser.add_argument("--tcfd_threshold",
                        default=0.7,
                        type=float,
                        help="The teacher confidence threshold for checking whether the sample x is teachable. i.e. teacher's cfd must both over this threshold.")
    parser.add_argument("--scfd_threshold",
                        default=0.6,
                        type=float,
                        help="The student confidence threshold for checking whether the sample x is teachable. i.ie student's cfd must lower that the threshold.")
    parser.add_argument("--r_t",
                        default=0.1,
                        type=float,
                        help="The addaptive rate of threshold for teacher clf after each iteration.")
    parser.add_argument("--r_s",
                        default=0.1,
                        type=float,
                        help="The addaptive rate of threshold for student clf after each iteration.")


    parser.add_argument("--save_subsample",
                        action='store_true',
                        help="Whether to save sub-sample set of L.")
    parser.add_argument("--sample_dir",
                        default='sub_data/',
                        type=str,
                        help="The dir that you save the sub-samples of L set.")
    parser.add_argument("--eval_dir",
                        default='"tri-models/eval_monitor/"',
                        type=str,
                        help="The dir that you save the eval of meta-tri-models.")
    parser.add_argument("--r",
                        default=0.7,
                        type=float,
                        help="The subset size of origin train set, e.g. 70 percent of the data (with replacement) for each sample.")
    parser.add_argument("--dataset",
                        default="isw",
                        type=str,
                        help="The prefix of train set L")
    args = parser.parse_args()

    if args.save_subsample:
        if os.path.exists(args.sample_dir) and os.listdir(args.sample_dir):
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.sample_dir))
        if not os.path.exists(args.sample_dir):
            os.makedirs(args.sample_dir)

        logger.info(" ***** Sub-sampling L ***** ")
        s1, s2, s3 = utils.random_subsample_replacement(r=args.r, dataset=args.dataset)
        logger.info(" Sampling with {} percents of origin train set:".format(args.r))
        logger.info(" S1 size = {}".format(len(s1)))
        logger.info(" S2 size = {}".format(len(s2)))
        logger.info(" S3 size = {}".format(len(s3)))
        logger.info(" Examples : {}".format(s1[:2]))
        joblib.dump(s1, '{}/train-isw-s1.pkl'.format(args.sample_dir))
        joblib.dump(s2, '{}/train-isw-s2.pkl'.format(args.sample_dir))
        joblib.dump(s3, '{}/train-isw-s3.pkl'.format(args.sample_dir))
        logger.info(" Save into : {}".format(args.sample_dir))
    else:
        # Start tri-training
        logger.info(" ***** Start Tri-training ***** ")
        # Start tri-training : save teachable samples
        tri_train = TriTraining(U=args.U, u=args.u, mi_dir=args.mi_dir, mj_dir=args.mj_dir, mk_dir=args.mk_dir, tcfd_threshold=args.tcfd_threshold, scfd_threshold=args.scfd_threshold, r_t=args.r_t, r_s=args.r_s, cos_score_threshold=args.cos_score_threshold)
        tri_train.fit(eval_dir=args.eval_dir)

if __name__ == '__main__':
    main()
