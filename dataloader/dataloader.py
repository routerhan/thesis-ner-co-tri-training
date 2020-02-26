import tensorflow as tf
import torch

from keras.preprocessing.sequence import pad_sequences

from preprocessor.preprocessor import Preprocessor
from transformers.tokenization_bert import BertTokenizer
from transformers.modeling_bert import BertModel, BertConfig


class DataLoader:
    """
    Implementation of converting data to tensor and load tensors into DataLoader
    """
    def __init__(self, filename = 'data/test-full-isw-release.tsv'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        self.preprocessor = Preprocessor(filename=filename)




    def load_bert_tokernizer(self, pretrained_model="bert-base-german-cased"):
        """
        :return : initialized tokenizer for preprocessing text from pretrained BERT model
        """
        tokenizer = BertTokenizer.from_pretrained(pretrained_model, do_lower_case=False)

        return tokenizer

    # return : Input IDs (tokens), tags (label IDs), attention masks
    def get_bert_dataset(self):
        # Initialize configuation
        MAX_LEN = 75
        bs = 32


        # Load the pretrained BERT tokenizer
        tokenizer =self.load_bert_tokernizer(pretrained_model="bert-base-german-cased")

        # Load preprocessed sentences, labels and tag2idx
        sentences = self.preprocessor.get_list_of_sentences()
        labels = self.preprocessor.get_list_of_nerlabels()
        tag2idx = self.preprocessor.get_tag2idx()

        # Tokenize the sentences
        tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

        # Get input id of token
        input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                            maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

        # Get tags of labels with id
        tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                        maxlen=MAX_LEN, value=tag2idx["O"], padding="post",
                        dtype="long", truncating="post")
