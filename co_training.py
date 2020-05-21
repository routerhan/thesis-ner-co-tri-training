import os
import utils
import logging
import joblib
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

	def __init__(self, modelA_dir:str, modelB_dir:str, top_n=5, k=3, u=40, save_preds=False, unlabel_dir=""):
		self.top_n = top_n
		self.k = k
		self.u = u
		self.clf_A = Ner(model_dir=modelA_dir)
		self.clf_B = Ner(model_dir=modelB_dir)
		self.modelA_dir = modelA_dir
		self.modelB_dir = modelB_dir

		# Cosine similarity score threshold for the level of agreement.
		self.cos_score_threshold = 0.7


	def prep_unlabeled_set(self, unlabel_dir):
		"""
		para : the dir of unlabeled data set
		return : list of sentences with index: [(1, 'I have apple'), (2, 'I am here'),..]
		"""
		sentences = joblib.load(unlabel_dir)
		sentences = [(i, sent) for i, sent in enumerate(sentences)]
		# mock small size
		sentences = sentences[:105] 
		return sentences

	def get_confident_preds(self, clf, unlabel_dir):
		"""
		This function will return confident predictions after iterations.
		
		Parameters: 

		return: 
		
		top_n_preds: [(index, sent, labels, confident_score), ...]
		"""
		U = self.prep_unlabeled_set(unlabel_dir)
		# TODO : give the unique index for sent...

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
		return top_n_preds

	def fit(self, save_preds:bool, unlabel_dir, label_dir=""):
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
		top_A = self.get_confident_preds(clf=self.clf_A, unlabel_dir=unlabel_dir)
		top_B = self.get_confident_preds(clf=self.clf_B, unlabel_dir=unlabel_dir)
		
		# Get agree preds between two models: identical check and confident_score
		compare_agree_list = self.get_agree_preds(predA=top_A, predB=top_B, save_agree=True)


		# Get agree preds between two models: identical check and confident_score
		# L_A, L_B = self.get_agree_preds(self, predA=top_A, predB=top_B, ignore_O=True, save_agree=False)

		# Extend train set....TODO: train data is in format of sentences and labels, as pkl...
		# Maybe write a finction to deal with it....
		# _, _, L_sents, L_labels = load_train_data(data_dir=label_dir)
		# extend_Eng_train_set()
		# extend_Gem_train_set()
		# Retrain

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
			with open(labeded_file, "w") as writer:
				for i, feature in enumerate(features):
					writer.write(str(feature.index)+'\n')
					writer.write(str(feature.sentence)+'\n')
					writer.write(str(feature.label)+'\n')
					writer.write(str(feature.avg_cfd_score)+'\n')
					writer.write('\n')
			writer.close()

		return features

	def get_agree_preds(self, predA, predB, save_agree=False, ignore_O=True):
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
		print("mutual_sent_ids", sorted(mutual_sent_ids))

		mutual_A = sorted([pred for pred in predA if pred[0] in mutual_sent_ids], key=lambda tup: tup[0], reverse=False)
		mutual_B = sorted([pred for pred in predB if pred[0] in mutual_sent_ids], key=lambda tup: tup[0], reverse=False)
		
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
			
			cos_score = utils.cosine_similarity(A_tag_list=A_tag_list, B_tag_list=B_tag_list)

			if cos_score > self.cos_score_threshold:
				compare_list.append((A_i, A_sent, A_label, A_score, B_sent, B_label, B_score, cos_score))
			else:
				pass
		print("agree_sent_ids", [i for (i, *tail) in compare_list ])
		
		# for (sent_index, A_tag_list, A_avg_cfd_score, B_tag_list, B_avg_cfd_score, cos_score) in compare_list:
		# 	if cos_score > 0.7:
		# 		# take one model as example
		# 		print('predA_sent: ', self.predA[i].sentence)
		# 		print('predA_label: ', self.predA[i].label)
		# 		print('predA_score: ', self.predA[i].avg_cfd_score)
		# 		print('')
		# 		print('predB_sent: ', self.predB[i].sentence)
		# 		print('predB_label: ', self.predB[i].label)
		# 		print('predB_score: ', self.predB[i].avg_cfd_score)
		# 		print('cos_score:', cos_score)
		# 		break

		if save_agree:
			with open("agree_results.txt", "w") as writer:
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
		return compare_list



# co_train = CoTraining(modelA_dir='models/', modelB_dir='test_model', save_preds=True, unlabel_dir='unlabel_sentences/2017_sentences.pkl')
# unlabeled_sents = co_train.prep_unlabeled_set(unlabel_dir='unlabel_sentences/2017_sentences.pkl')
# print('U set:', unlabeled_sents[:3])

# i = 32
# # take one model as example
# print('predA_sent: ', co_train.predA[i].sentence)
# print('predA_label: ', co_train.predA[i].label)
# print('predA_score: ', co_train.predA[i].avg_cfd_score)

# print('\n')
# print('predB_sent: ', co_train.predB[i].sentence)
# print('predB_label: ', co_train.predB[i].label)
# print('predB_score: ', co_train.predB[i].avg_cfd_score)

# print("get_agree_preds")
# co_train.get_agree_preds(save_agree=True)

co_train = CoTraining(modelA_dir='models/', modelB_dir='test_model', save_preds=True, unlabel_dir='unlabel_sentences/2017_sentences.pkl')
compare_agree_list = co_train.fit(unlabel_dir='unlabel_sentences/2017_sentences.pkl', save_preds=True)
