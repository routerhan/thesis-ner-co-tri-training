import os
import utils
import logging
import joblib
import html
import re
import json
from predict import Ner
from run_ner import load_train_data

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class TriTraining:
    """
    This class implements teacher-student tri-training algorithm for NER task.

    Parameters: 
    L - set of labeled samples, 
    U - set of unlabeled samples, 
    mi, j,k - teacher-student models, 
    threshold_t - teacher threshold,
    threshold_s - student threshold, 
    r_t,r_s - teacher-student adaptive rates
    """

    def __init__(self, U, u, mi_dir, mj_dir, mk_dir, tcfd_threshold, scfd_threshold, r_t, r_s, cos_score_threshold):
        # U : unlabeled list of sentences
        self.U = U
        # pool value of unlabeled samples.
        self.u = u
        self.mi = (Ner(model_dir=mi_dir), mi_dir)
        self.mj = (Ner(model_dir=mj_dir), mj_dir)
        self.mk = (Ner(model_dir=mk_dir), mk_dir)
        self.tcfd_threshold = tcfd_threshold
        self.scfd_threshold = scfd_threshold
        self.r_t = r_t
        self.r_s = r_s
        self.cos_score_threshold = cos_score_threshold
        self.save_teachable = True

    def is_teachable(self, tm1, tm2, sm):
        """
        input : teachers and student preds, e.g. (sentence, label, avg_cfd_score)
        return : boolean, the labeled identified teachable samples Li, which will be added into origin L.
        """
        
        # Identitiy check : cos similarity score for tag_list
        t1_labels = tm1[1]
        t2_labels = tm2[1]
        check = False
        cos_score = utils.cosine_similarity(A_tag_list=t1_labels, B_tag_list=t2_labels)
        if cos_score > self.cos_score_threshold:
            tcfd = min(tm1[2], tm2[2])
            scfd = sm[2]
            if tcfd > self.tcfd_threshold and scfd<self.scfd_threshold:
                check = True
        return check

    def assign_teacher_student_by_e(self, valid_on='test'):
        """
        Initialize the teacher and student roles by estimating the "error rate", validated by the test-label set.
        Pair-wise comparison: (mi, mj), (mi, mk), (mj, mk)
        return : <t-model1, t-model2, s-model>
        """
        val_sents = joblib.load('data/{}-isw-sentences.pkl'.format(valid_on))[:1000]
        val_labels = joblib.load('data/{}-isw-labels.pkl'.format(valid_on))[:1000]
        assert len(val_sents) == len(val_labels)

        # Ingore O for similarity check...
        val_labels = [[tag for tag in tags if tag != 'O'] for tags in val_labels]
        # print("val_labels", val_labels[:3])
        # [['B-NRP', 'B-NRP'], [], ['B-DATE'], [], [], []]

        label_ls_i = []
        label_ls_j = []
        label_ls_k = []
        for sent in val_sents:
            predi=(self.mi[0]).predict(text=sent)
            predj=(self.mj[0]).predict(text=sent)
            predk=(self.mk[0]).predict(text=sent)

            labeli = [dic['tag'] for dic in predi]
            labelj = [dic['tag'] for dic in predj]
            labelk = [dic['tag'] for dic in predk]
            # print("labeli", labeli)
            # ['O', 'O', 'O', 'O', 'O', 'B-NRP', 'O', 'O', 'B-NRP']

            label_ls_i.append([tag for tag in labeli if tag != 'O'])
            label_ls_j.append([tag for tag in labelj if tag != 'O'])
            label_ls_k.append([tag for tag in labelk if tag != 'O'])
            # print("label_ls_i", label_ls_i)
            # [['B-NRP', 'B-NRP'], [], ['B-DATE'], [], [], []]

        # [(labeli, labelj) for labeli, labelj in zip(label_i, label_j) if cosine_similarity(labeli, labelj)>0.99]
        common_label_ij = [(labeli, labelj) for labeli, labelj in zip(label_ls_i, label_ls_j) if utils.cosine_similarity(labeli, labelj)>= self.cos_score_threshold]
        common_label_ik = [(labeli, labelk) for labeli, labelk in zip(label_ls_i, label_ls_k) if utils.cosine_similarity(labeli, labelk)>= self.cos_score_threshold]
        common_label_jk = [(labelj, labelk) for labelj, labelk in zip(label_ls_j, label_ls_k) if utils.cosine_similarity(labelj, labelk)>= self.cos_score_threshold]
        # print("common_label_ij", common_label_ij)
        
        # Comapre common labels with true labels to compute the error rate in further steps.
        w_label_ij = [(labeli, labelj, t) for labeli, labelj, t in zip(label_ls_i,label_ls_j, val_labels) if utils.cosine_similarity(labeli, labelj)>= self.cos_score_threshold and utils.cosine_similarity(labeli, t)<self.cos_score_threshold and utils.cosine_similarity(labelj, t)<self.cos_score_threshold]
        w_label_ik = [(labeli, labelk, t) for labeli, labelk, t in zip(label_ls_i,label_ls_k, val_labels) if utils.cosine_similarity(labeli, labelk)>= self.cos_score_threshold and utils.cosine_similarity(labeli, t)<self.cos_score_threshold and utils.cosine_similarity(labelk, t)<self.cos_score_threshold]
        w_label_jk = [(labelj, labelk, t) for labelj, labelk, t in zip(label_ls_j,label_ls_k, val_labels) if utils.cosine_similarity(labelj, labelk)>= self.cos_score_threshold and utils.cosine_similarity(labelj, t)<self.cos_score_threshold and utils.cosine_similarity(labelk, t)<self.cos_score_threshold]
        # print("w_label_ij", w_label_ij)

        # Compute the error pair-wisely
        e_ij = round(len(w_label_ij) / len(common_label_ij), 4)
        e_ik = round(len(w_label_ik) / len(common_label_ik), 4)
        e_jk = round(len(w_label_jk) / len(common_label_jk), 4)

        # print("w_label_ij", w_label_ij[:2])
        # print("len of common_label_ij:", len(common_label_ij))
        # print("len of w_label_ij:", len(w_label_ij))
        # print(common_label_ij[:2])
        # print(w_label_ij[:2])

        logger.info(" ***** Assigning teacher and student clf ***** ")
        logger.info(" e_ij : {}".format(e_ij))
        logger.info(" e_ik : {}".format(e_ik))
        logger.info(" e_jk : {}".format(e_jk))

        e_rate = 0
        t1, t2, s = (1, 1, 1)
        if min(e_ij, e_ik, e_jk)==e_ij:
            logger.info(" teacher models are assign to : mi, mj")
            e_rate = e_ij
            t1, t2, s = (self.mi, self.mj, self.mk)
        elif min(e_ij, e_ik, e_jk)==e_ik:
            logger.info(" teacher models are assign to : mi, mk")
            e_rate = e_ik
            t1, t2, s = (self.mi, self.mk, self.mj)
        elif min(e_ij, e_ik, e_jk)==e_jk:
            logger.info(" teacher models are assign to : mj, mk")
            e_rate = e_jk
            t1, t2, s = (self.mj, self.mk, self.mi)
        return t1, t2 , s, e_rate

    def fit(self, eval_dir="tri-models/eval_monitor/"):
        """
        This function takes care of tri-training process.
        """
        # Load unlabel set
        U = utils.prep_unlabeled_set(self.U)
        # (Ner(model_dir=mi_dir), mi_dir), i.e. self.mi[0]=Ner(model_dir=mi_dir), self.mi[1]= mi_dir
        role_rotate = [(self.mi, self.mj, self.mk), (self.mj, self.mk, self.mi), (self.mi, self.mk, self.mj)]

        it = 0
        while self.tcfd_threshold >= self.scfd_threshold:
            it += 1
            ext_data_s = []

            # U_ is for classifier to choose, we take it from back
            U_ = U[-min(len(U), self.u):]
            U = U[:-len(U_)]
            # print("len of U_:{}".format(len(U_)))
            # Save teachable samples of each clf candidates in their model_dir.
            for (t1, t2, s) in role_rotate:
                s_dir = s[1]
                # Get preds and check if the sample x in U_ is teachable
                teachable_preds=[]
                for (i, x) in U_:
                    try:
                        compare_ts_list = []
                        for model in [t1[0], t2[0], s[0]]:
                            pred = model.predict(text=x)
                            sentence = [dic['word'] for dic in pred]
                            label = [dic['tag'] for dic in pred]
                            avg_cfd_score = utils.get_avg_confident_score(pred_result=pred, ignore_O=True)
                            compare_ts_list.append((sentence ,label, avg_cfd_score))
                        t1_preds, t2_preds, s_preds = compare_ts_list

                        # Check if x is teachable base on its prediction confident score.
                        if self.is_teachable(tm1=t1_preds, tm2=t2_preds, sm=s_preds):
                            ext_sent = []
                            assert t1_preds[0]==t2_preds[0]
                            sentence=t1_preds[0]
                            ext_sent.append(" ".join(sentence))
                            teachable_preds.append((ext_sent, t1_preds, t2_preds, s_preds))
                    except:
                        pass
                logger.info(" Current teacher threshold: {}, student threshold: {}".format(self.tcfd_threshold,self.scfd_threshold))
                logger.info(" ***** Picking teachable samples ***** ")
                logger.info(" num of teachable for {} = {}".format(s[1],len(teachable_preds)))
                logger.info(" ***** Example samples ***** ")
                logger.info(" Sent = {}".format(teachable_preds[0][0]))
                logger.info(" t1 preds = {}".format(teachable_preds[0][1][1]))
                logger.info(" t2 preds = {}".format(teachable_preds[0][2][1]))
                logger.info(" s preds = {}".format(teachable_preds[0][3][1]))

                # Save this txt file is used for analysis
                if self.save_teachable:
                    analysis_file = os.path.join(s_dir, "{}-{}-analysis-samples.txt".format(it ,len(teachable_preds)))
                    with open(analysis_file, "w", encoding="utf-8") as w:
                        for features in teachable_preds:
                            sent, (s, t1_label, t1cfd), (s, t2_labels, t2cfd), (s, s_labels, scfd)=features
                            w.write(str(sent)+'\n')
                            w.write(str(t1_label)+'\t')
                            w.write(str(t1cfd)+'\n')
                            w.write(str(t2_labels)+'\t')
                            w.write(str(t2cfd)+'\n')
                            w.write(str(s_labels)+'\t')
                            w.write(str(scfd)+'\n')
                            w.write('\n')
                    w.close()
                    logger.info("Save analysis samples under student model dir: {}".format(analysis_file))

                # Save as pkl for further re-training process, saved in student dir
                ext_sents = [sent[0] for (sent, _, _, _) in teachable_preds]
                ext_labels = []
                for (_, (_, t1_label, t1_cfd), (_, t2_label, t2_cfd),*_) in teachable_preds:
                    if t1_label==t2_label:
                        ext_labels.append(t1_label)
                    elif t1_cfd > t2_cfd:
                        ext_labels.append(t1_label)
                    else:
                        ext_labels.append(t2_label)
                # 1_108_ext_sents.pkl
                prefix = "{}_{}".format(it, len(ext_sents))
                assert len(ext_sents) == len(ext_labels)
                joblib.dump(ext_sents,'{}{}_ext_sents.pkl'.format(s_dir, prefix))
                joblib.dump(ext_labels,'{}{}_ext_labels.pkl'.format(s_dir, prefix))
                logger.info("Save ext teachable sentences, length:{}".format(len(ext_sents)))
                logger.info("Save sents and labels as pkl file in dir: {}".format(s_dir))
                ext_data_s.append((s_dir, len(ext_sents)))

                # Save the tri-training config
                tritrain_config = {
                    "Ext data in student dir": s_dir, 
                    "Teacher 1": t1[1], 
                    "Teacher 2": t2[1],
                    "Student": s_dir,
                    "U": self.U,
                    "Pool value u": self.u,
                    "tcfd_threshold": self.tcfd_threshold,
                    "scfd_threshold": self.scfd_threshold,
                    "Adaptive r_t": self.r_t,
                    "Adaptive r_s": self.r_s, 
                    "Similarity threshold": self.cos_score_threshold, 
                    "Num of teachable_preds": len(ext_sents),
                    "Prefix": prefix
                    }

                json.dump(tritrain_config, open(os.path.join(s_dir, "{}_tri_config.json".format(it)), 'w'))
                logger.info("Save tri-training config in dir: {}".format(s_dir))

            # Once each of the three classifiers has been rotated to be t1, t2 and s,
            # start to retrain s model on the expanded dataset respectively.
            # ext_data_s = [(tri-models/s3_model/, 108), (tri-models/s1_model/, 96), (tri-models/s2_model/, 87)]
            for i in range(len(ext_data_s)):
                tmp_ext_data_dir = ext_data_s[i][0]
                prefix = ""
                if tmp_ext_data_dir.find("s3") != -1:
                    prefix = "s3"
                    ori_s_dir = "sub_data/train-isw-s3.pkl"
                elif tmp_ext_data_dir.find("s1") != -1:
                    prefix = "s1"
                    ori_s_dir = "sub_data/train-isw-s1.pkl"
                else:
                    prefix = "s2"
                    ori_s_dir = "sub_data/train-isw-s2.pkl"
                logger.info("***** Retraining each student models : {} *****".format(tmp_ext_data_dir))
                ext_model_saved_dir = "tri-ext-models"
                ext_output_dir = "{}/{}_ext_{}_model/".format(ext_model_saved_dir, it ,prefix)
                # If there is a ext_model exsist in previous iteration, remove it and save the new_ext_model of this iteration.
                # if os.path.exists(ext_output_dir) and os.listdir(ext_output_dir):
                #     os.system("rm {}*".format(ext_output_dir))
                #     os.system("rmdir {}".format(ext_output_dir))
                    # extend again the ext_train_set in previous iteration, e.g. 7806+16+18+20
                if it != 1:
                    ori_s_dir = ori_s_dir[:9]+"ext-"+ori_s_dir[9:]
                    logger.info("***** Using previous ext_train_set : {} *****".format(ori_s_dir))
                script = "python run_ner.py --max_seq_length 128 --do_train --do_subtrain --extend_L_tri --it {} --subtrain_dir {} --ext_data_dir {} --ext_output_dir {}".format(it, ori_s_dir, tmp_ext_data_dir, ext_output_dir)
                os.system(script)
                # Save the eval result of each student re-trained model
                it_prefix = "{}_{}".format(it, prefix)
                script = "python run_ner.py --do_eval --eval_on test --extend_L_tri --eval_dir {} --ext_output_dir {} --it_prefix {}".format(eval_dir, ext_output_dir, it_prefix)
                os.system(script)
            # Assign ext_model as new rotated candidates roles
            self.mi = (Ner(model_dir="{}/{}_ext_s1_model/".format(ext_model_saved_dir, it)), "{}/{}_ext_s1_model/".format(ext_model_saved_dir, it))
            self.mj = (Ner(model_dir="{}/{}_ext_s2_model/".format(ext_model_saved_dir, it)), "{}/{}_ext_s2_model/".format(ext_model_saved_dir, it))
            self.mk = (Ner(model_dir="{}/{}_ext_s3_model/".format(ext_model_saved_dir, it)), "{}/{}_ext_s3_model/".format(ext_model_saved_dir, it))
            role_rotate = [(self.mi, self.mj, self.mk), (self.mj, self.mk, self.mi), (self.mi, self.mk, self.mj)]
            # Adjust teacher and student threshold as the knowledge gap is smaller
            self.tcfd_threshold = self.tcfd_threshold - self.r_t
            self.scfd_threshold = self.scfd_threshold + self.r_s
            logger.info(" Adapted teacher threshold: {}, student threshold: {}".format(self.tcfd_threshold,self.scfd_threshold))
            logger.info("***** End of iteration : {} *****".format(it))

