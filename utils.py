import logging
import joblib
from sklearn.model_selection import train_test_split
from collections import Counter
from preprocessor import IswPreprocessor

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def split_data(data_dir='data/full-isw-release.tsv'):
    """
    This function will split the full dataset into `train`, `dev`, `test` and save the results.

    return : the num of labels on full-data-set (not only train data or dev data)
    """

    if "isw" in str(data_dir):
        dataset = "isw"
        pre = IswPreprocessor(filename=data_dir)

    sentences = pre.sentences
    labels = pre.labels
    label_list = pre.get_labels()
    num_labels = len(label_list) + 1

    logger.info("***** Train/dev/test split: 70/20/10 *****")
    X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.1, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size= 2/9, random_state=1) # 2/9 x 0.9 = 0.2
    logger.info("  Num train = %d", len(X_train))
    logger.info("  Num dev = %d", len(X_val))
    logger.info("  Num test = %d", len(X_test))
    logger.info("***** Save as pkl in /data *****")
    joblib.dump(X_train,'data/train-{}-sentences.pkl'.format(dataset))
    joblib.dump(y_train,'data/train-{}-labels.pkl'.format(dataset))
    joblib.dump(X_test,'data/test-{}-sentences.pkl'.format(dataset))
    joblib.dump(y_test,'data/test-{}-labels.pkl'.format(dataset))
    joblib.dump(X_val,'data/dev-{}-sentences.pkl'.format(dataset))
    joblib.dump(y_val,'data/dev-{}-labels.pkl'.format(dataset))

    return label_list, num_labels

def cosine_similarity(A_tag_list:list, B_tag_list:list):
    """
    pred_result = [ ['O', 'O', 'O', 'B-GPE', 'O', 'B-TIME'], [] ...  ]
    return: cosine similarity score for identical check
    """
    a_vals = Counter(A_tag_list)
    b_vals = Counter(B_tag_list)
    # convert to word-vectors
    words  = list(a_vals.keys() | b_vals.keys())
    a_vect = [a_vals.get(word, 0) for word in words]
    b_vect = [b_vals.get(word, 0) for word in words]
    # find cosine
    len_a  = sum(av*av for av in a_vect) ** 0.5
    len_b  = sum(bv*bv for bv in b_vect) ** 0.5
    dot    = sum(av*bv for av,bv in zip(a_vect, b_vect))
    try:
        cosine = dot / (len_a * len_b)
    except:
        cosine = 0
    return round(cosine,4)

def get_avg_confident_score(pred_result:list, ignore_O=True):
    """
    pred_result = [  {'confidence': 0.2621215581893921, 'tag': 'O', 'word': 'ich'},
                {'confidence': 0.0977315902709961, 'tag': 'I-SORD', 'word': 'aus'},
                {'confidence': 0.1431599259376526, 'tag': 'O', 'word': 'EU'}  ]
    return: avg score of every confidence score in result
    """
    if ignore_O:
        scores = [dic['confidence'] for dic in pred_result if dic['tag'] != 'O']
    else:
        scores = [dic['confidence'] for dic in pred_result]
        
    try:
        avg_score = sum(scores)/len(scores)
    except:
        avg_score = 0
    return round(avg_score, 4)

def get_hyperparameters(model, ff):

    # ff: full_finetuning
    if ff:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.0,
            },
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    return optimizer_grouped_parameters
