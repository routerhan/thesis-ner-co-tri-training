import joblib
import preprocessor
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def split_data(data_dir='data/full-isw-release.tsv'):

    if "isw" in str(data_dir):
        dataset = "isw"
        pre = preprocessor.IswPreprocessor(filename=data_dir)

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

# if __name__ == "__main__":
#     split_data()