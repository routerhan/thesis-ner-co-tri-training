import utils
import pickle


class CoTraining:
    """
    This class takes care of the implementation of co-training algorithm

	Parameters:
	clf1 - The classifier that will be used in the cotraining algorithm on the X1 feature set
		(Note a copy of clf will be used on the X2 feature set if clf2 is not specified).
	clf2 - A different classifier type can be specified to be used on the X2 feature set
		 if desired.
	top_n - The number of the most confident examples that will be 'labeled' by each classifier during each iteration
	k -  The number of iterations. The default is 30 
	u - The size of the pool of unlabeled samples from which the classifier can choose
		Default - 75 
	"""

    def __init__(self, clf1, clf2, top_n, k, u):
        pass