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

class PredictFeatures:
	"""
	a set of predcition
	"""
	def __init__(self, index, sentence, label, avg_cfd_score):
		self.index = index
		self.sentence = sentence
		self.label = label
		self.avg_cfd_score = avg_cfd_score

class CoTraining:
	"""
	This class takes care of the implementation of co-training algorithm

	Parameters:
	modelA_dir - The dir of pre-trained model that will be used in the cotraining algorithm on the X1 feature set
	modelB_dir - The dir of another pre-trained model can be specified to be used on the X2 feature set.
	top_n - The number of the most confident examples that will be 'labeled' by each classifier during each iteration
	k -  The number of iterations. The default is 30 
	u - The size of the pool of unlabeled samples from which the classifier can choose. Default - 75 
	"""

	def __init__(self, modelA_dir:str, modelB_dir:str, top_n=5, k=3, u=40, save_preds=False):
		self.top_n = top_n
		self.k = k
		self.u = u
		self.clf_A = Ner(model_dir=modelA_dir)
		self.clf_B = Ner(model_dir=modelB_dir)
		self.modelA_dir = modelA_dir
		self.modelB_dir = modelB_dir

		# Cosine similarity score threshold for the level of agreement.
		self.cos_score_threshold = 0.9

	# Load txt file as: 1. de_sents.txt, 2. en_sents.txt ..
	def prep_unlabeled_set(self, unlabel_dir):
		"""
		para : the dir of unlabeled data set
		return : list of sentences with index: [(1, 'I have apple'), (2, 'I am here'),..]
		"""
		file = open(unlabel_dir, "r", encoding="utf-8")
		sentences = []
		for i, sent in enumerate(file):
			sent=sent.strip()
			sent=html.unescape(sent)
			sentences.append((i, sent)) 
		return sentences

	def get_confident_preds(self, clf, unlabel_dir, model_dir, save_preds=False):
		"""
		This function will return confident predictions after iterations.
		
		Parameters: 

		return: 
		
		top_n_preds: [(index, sent, labels, confident_score), ...]
		"""
		U = self.prep_unlabeled_set(unlabel_dir)

		top_n_preds = []
		it = 0
		while it != self.k:
			it += 1

			# U_ is for classifier to choose, we take it from back
			U_ = U[-min(len(U), self.u):]
			len_U_ = len(U_)

			# Remove U_ from U
			U = U[:-len(U_)]

			# Get pred form model, append preds to pred_features each iteration.
			preds = []
			for (i, sent) in U_:
				pred = clf.predict(sent)
				logging.info('Model - it:{}, tagging: {} / {}'.format(it, i, len_U_))
				# [  {'confidence': 0.2621215581893921, 'tag': 'O', 'word': 'ich'},...]
				sentence = [dic['word'] for dic in pred]
				label = [dic['tag'] for dic in pred]
				avg_cfd_score = utils.get_avg_confident_score(pred_result=pred, ignore_O=True)
				preds.append((i, sentence, label, avg_cfd_score))
			
			temp_top_preds = sorted(preds, key=lambda tup: tup[3], reverse=True)[:self.top_n]
			[top_n_preds.append(tA) for tA in temp_top_preds]
		
		if save_preds:
			labeded_file = os.path.join(model_dir, "labeled_results.txt")
			with open(labeded_file, "w", encoding="utf-8") as writer:
				for (i, sentence, label, avg_cfd_score) in top_n_preds:
					writer.write(str(i)+'\n')
					writer.write(str(sentence)+'\n')
					writer.write(str(label)+'\n')
					writer.write(str(avg_cfd_score)+'\n')
					writer.write('\n')
			writer.close()

		return top_n_preds

	def fit(self, save_preds:bool, save_agree:bool, ext_output_dir, de_unlabel_dir, en_unlabel_dir, label_dir=""):
		"""
		This func will execute co-training algorithm.

		Parameters:
		U 			- the unlabeled dataset, the amount will change dynamic in each iteraion.
		U_			- the subset of U for each iteraion, the amount is decided by the value of u.
		L			- the original label dataset. e.g. isw-dataset for German, OntoNote 5.0 for English
		L_A 		- top_n confident result of model A, which agreed with model B.
		L_B 		- top_n confident result of model B, which agreed with model A.
		ext_L_A 	- new extended labeled set for training a new model A, based on feature A.
		ext_L_B		- new extended labeled set for training a new model B, based on feature B.
		"""

		# Get top_n preds of each iteraions, which should be ready to be used for extending L.
		top_A = self.get_confident_preds(clf=self.clf_A, unlabel_dir=de_unlabel_dir, save_preds=True, model_dir=self.modelA_dir)
		top_B = self.get_confident_preds(clf=self.clf_B, unlabel_dir=en_unlabel_dir, save_preds=True, model_dir=self.modelB_dir)

		# The sentences should be the same but in different languages. => cross-lingual features.
		assert len(top_A) == len(top_B)
		
		# Get agree preds between two models: identical check and confident_score
		
		compare_agree_list, ext_L_A_sents, ext_L_A_labels, ext_L_B_sents, ext_L_B_labels = self.get_agree_preds(predA=top_A, predB=top_B, save_agree=True, ext_output_dir=ext_output_dir)

		joblib.dump(ext_L_A_sents,'{}/{}_ext_L_A_sents.pkl'.format(ext_output_dir, len(ext_L_A_sents)))
		logger.info("Save ext de sentences, length:{}".format(len(ext_L_A_sents)))

		joblib.dump(ext_L_A_labels,'{}/{}_ext_L_A_labels.pkl'.format(ext_output_dir, len(ext_L_A_labels)))
		logger.info("Save ext de labels, length:{}".format(len(ext_L_A_labels)))

		joblib.dump(ext_L_B_sents,'{}/{}_ext_L_B_sents.pkl'.format(ext_output_dir, len(ext_L_B_sents)))
		logger.info("Save ext en sentences, length:{}".format(len(ext_L_B_sents)))

		joblib.dump(ext_L_B_labels,'{}/{}_ext_L_B_labels.pkl'.format(ext_output_dir, len(ext_L_B_labels)))
		logger.info("Save ext en labels, length:{}".format(len(ext_L_B_labels)))

		cotrain_config={
			"ext_output_dir": ext_output_dir,
            "Approach":"Cross-lingual Co-training",
            "Model A de":self.modelA_dir,
            "Model B en":self.modelB_dir,
            "Pool value u":self.u,
            "Confident top_n":self.top_n,
            "Iteration k":self.k,
            "Agree threshold cos_score":self.cos_score_threshold,
			"Ext number of L_" : len(ext_L_A_sents),
			"Prefix": len(ext_L_A_sents)
			}
		json.dump(cotrain_config, open(os.path.join(ext_output_dir, "cotrain_config.json"), 'w'))
		return compare_agree_list


	def get_preds(self, model_dir:str, save_preds:bool, unlabel_dir):
		"""
		This function will call a model to make prediction and save those prediction as class:PredictFeatures.
		return : pred_features : PredictFeatures(sentences=sentences[], labels=labels[], avg_cfd_score)
		"""
		model = Ner(model_dir=model_dir)
		unlabeled_sents = self.prep_unlabeled_set(unlabel_dir)
		features = []
		num_unlabeled_sents = len(unlabeled_sents)
		for i ,text in enumerate(unlabeled_sents):
			output = model.predict(text=text)
			logging.info('Model: {}, tagging: {} / {}'.format(model_dir ,i, num_unlabeled_sents))
			# [  {'confidence': 0.2621215581893921, 'tag': 'O', 'word': 'ich'},...]
			sentence = [dic['word'] for dic in output]
			label = [dic['tag'] for dic in output]
			avg_cfd_score = utils.get_avg_confident_score(pred_result=output, ignore_O=True)

			features.append(PredictFeatures(index=i, sentence=sentence, label=label, avg_cfd_score=avg_cfd_score))
		
		if save_preds:
			labeded_file = os.path.join(model_dir, "labeled_results.txt")
			with open(labeded_file, "w", encoding="utf-8") as writer:
				for i, feature in enumerate(features):
					writer.write(str(feature.index)+'\n')
					writer.write(str(feature.sentence)+'\n')
					writer.write(str(feature.label)+'\n')
					writer.write(str(feature.avg_cfd_score)+'\n')
					writer.write('\n')
			writer.close()

		return features
	
	def multiple_replace(self, text):
		# Create a regular expression  from the dictionary keys
		dict={
			"EVENT" : "EVT",
			"LANGUAGE" : "LAN",
			"MONEY" : "MON",
			"NORP" : "NRP",
			"PERSON" : "PER",
			"PERCENT" : "PERC",
			"QUANTITY" : "QUAN",
			"WORK_OF_ART" :"ART"
		}
		regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))
		# For each match, look-up corresponding value in dictionary
		return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)

	def get_agree_preds(self, predA, predB, save_agree=False, ignore_O=True, ext_output_dir=""):
		"""
		return : a tuple both-agreed confident predicted data from predA and predB.
		-both-agreed -> cosine similarity score, for multi-labels comparison.
		-confident -> by averaging the confidence socre of each tokens, for multi-labels confident check.

		e.g. compare_agree_list = [(sent_index, A_tag_list, A_avg_cfd_score, B_tag_list, B_avg_cfd_score, cos_score), ()...]
		Parameters:
		predA : [(1, ["this", "is"], ["O", "O"]), (), ...]

		Agreement based on : 
		1. identical check : utils.cosine_similarity
		2. confident threshold : PredictFeatures.avg_cfd_score
		"""
		assert len(predA) == len(predB)
		# Get sent ids of each model, and find the common ones.
		A_id_list = [i for (i, _, _, _)in predA ]
		B_id_list = [i for (i, _, _, _)in predB ]
		mutual_sent_ids = set(A_id_list) & set(B_id_list)
		# print("mutual_sent_ids", sorted(mutual_sent_ids))

		mutual_A = sorted([pred for pred in predA if pred[0] in mutual_sent_ids], key=lambda tup: tup[0], reverse=False)
		mutual_B = sorted([pred for pred in predB if pred[0] in mutual_sent_ids], key=lambda tup: tup[0], reverse=False)
		
		# There are the extended L_ set for adding the original train set.
		ext_L_A_sents=[]
		ext_L_A_labels=[]
		ext_L_B_sents=[]
		ext_L_B_labels=[]

		compare_list = []
		for (A_i, A_sent, A_label, A_score), (B_i, B_sent, B_label, B_score) in zip(mutual_A, mutual_B):
			# index of sent should be the same
			assert A_i == B_i
			# ignore 'O'
			if ignore_O:
				A_tag_list = [tag for tag in A_label if tag != 'O']
				B_tag_list = [tag for tag in B_label if tag != 'O']
			else:
				A_tag_list = A_label
				B_tag_list = B_label
			
			# Map NER tag , e.g. PERSON -> PER
			B_tag_list = [self.multiple_replace(tag) for tag in B_tag_list]
			cos_score = utils.cosine_similarity(A_tag_list=A_tag_list, B_tag_list=B_tag_list)

			if cos_score > self.cos_score_threshold:
				ext_L_A_sents.append(" ".join(A_sent))
				ext_L_B_sents.append(" ".join(B_sent))
				ext_L_A_labels.append(A_label)
				ext_L_B_labels.append(B_label)
				
				compare_list.append((A_i, A_sent, A_label, A_score, B_sent, B_label, B_score, cos_score))
			else:
				pass
		# print("agree_sent_ids", [i for (i, *tail) in compare_list ])
		# Number of extended sents should be the same as two models
		assert len(ext_L_A_sents) == len(ext_L_B_sents)

		if save_agree:
			with open("{}/agree_results.txt".format(ext_output_dir), "w", encoding="utf-8") as writer:
				for feature in compare_list:
					A_i, A_sent, A_label, A_cfd_score, B_sent, B_label, B_cfd_score, cos_score = feature
					writer.write(str(A_i)+'\n')
					writer.write(str(A_sent) + '\t')
					writer.write(str(A_label) + '\t')
					writer.write(str(A_cfd_score)+'\n')
					writer.write(str(B_sent) + '\t')
					writer.write(str(B_label) + '\t')
					writer.write(str(B_cfd_score) + '\n')
					writer.write(str(cos_score) + '\n')
					writer.write('\n')
			writer.close()
		return compare_list, ext_L_A_sents, ext_L_A_labels, ext_L_B_sents, ext_L_B_labels


# co_train = CoTraining(modelA_dir='baseline_model/', modelB_dir='onto_model/', save_preds=True)
# compare_agree_list = co_train.fit(ext_output_dir='ext_data', de_unlabel_dir='machine_translation/2017_de_sents.txt', en_unlabel_dir='machine_translation/2017_en_sents.txt', save_agree=True, save_preds=True)
