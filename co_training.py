import os
import utils
import logging
import joblib
from predict import Ner

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
		self.predA = self.get_preds(model_dir=modelA_dir, save_preds=save_preds, unlabel_dir=unlabel_dir)
		self.predB = self.get_preds(model_dir=modelB_dir, save_preds=save_preds, unlabel_dir=unlabel_dir)


	def prep_unlabeled_set(self, unlabel_dir):
		"""
		para : the dir of unlabeled data set
		return : list of sentences : ['I have apple', 'I am here', 'hello ']
		"""
		sentences = joblib.load(unlabel_dir)
		# mock small size
		sentences = sentences[:20] 
		return sentences

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

	def get_agree_preds(self, save_agree=False, ignore_O=True):
		"""
		return : a tuple both-agreed confident predicted data from predA and predB, which is ready for expansion of L.
		e.g. 

		Agreement based on : 
		1. identical check : utils.cosine_similarity
		2. confident threshold : PredictFeatures.avg_cfd_score
		"""
		predicted_dataA = self.predA
		predicted_dataB = self.predB
		assert len(predicted_dataA) == len(predicted_dataB)
		
		compare_list = []
		for featureA, featureB in zip(predicted_dataA, predicted_dataB):
			assert featureA.index == featureB.index
			# ignore 'O'
			if ignore_O:
				A_tag_list = [tag for tag in featureA.label if tag != 'O']
				B_tag_list = [tag for tag in featureB.label if tag != 'O']
			else:
				A_tag_list = featureA.label
				B_tag_list = featureB.label
			
			cos_score = utils.cosine_similarity(A_tag_list=A_tag_list, B_tag_list=B_tag_list)

			compare_list.append((featureA.index, A_tag_list, featureA.avg_cfd_score, B_tag_list, featureB.avg_cfd_score, cos_score))
		
		for (sent_index, A_tag_list, A_avg_cfd_score, B_tag_list, B_avg_cfd_score, cos_score) in compare_list:
			if cos_score > 0.7:
				i = sent_index
				# take one model as example
				print('predA_sent: ', self.predA[i].sentence)
				print('predA_label: ', self.predA[i].label)
				print('predA_score: ', self.predA[i].avg_cfd_score)
				print('')
				print('predB_sent: ', self.predB[i].sentence)
				print('predB_label: ', self.predB[i].label)
				print('predB_score: ', self.predB[i].avg_cfd_score)
				print('cos_score:', cos_score)
				break

		if save_agree:
			with open("agree_results.txt", "w") as writer:
				for feature in compare_list:
					index, A_label, A_cfd_score, B_label, B_cfd_score, cos_score = feature
					writer.write(str(index)+'\n')
					writer.write(str(A_label) + '\t')
					writer.write(str(A_cfd_score)+'\n')
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
