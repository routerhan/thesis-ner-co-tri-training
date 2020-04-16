import logging

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask

def convert_examples_to_features(all_sentences, all_labels, label_list, max_seq_length, tokenizer):
    """
    :input: examples
    :return: list of features
    :input_ids, input_mask, segment_ids, label_ids, valid_ids, label_mask
    """
    label_map = {label : i for i, label in enumerate(label_list, 1)}
    print('hi label_map', label_map)
    print('\n')

    features = []
    for index, sentence in enumerate(all_sentences):
        textlist = sentence.split(' ')
        labellist = all_labels[index]

        tokens = []
        labels = []
        valid = []
        label_mask = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)
        if len(tokens)>= max_seq_length -1 :
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0,1)
        label_mask.insert(0,1)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        if index < 5:
            logger.info("*** Features Example ***")
            logger.info("textlist: %s" % " ".join([str(x) for x in textlist]))
            logger.info("labellist: %s" % " ".join([str(x) for x in labellist]))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info(len(tokens))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("valid: %s" % " ".join([str(x) for x in valid]))
            logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
            logger.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_ids))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask))
    return features


class OntoPreprocessor:
    def __init__(self, filename='data/onto.train.ner.sample'):
        logger.info('------ Preprocssing OntoNote English corpus ------')
        self.file = open(filename, encoding='utf-8')
        self.sentences, self.labels, self.flat_labels = self.get_sentences_and_labels()

        logger.info("Number of sentences: {0} ".format(len(self.sentences)))
        logger.info("Number of tags: {0} ".format(len(self.get_labels())))
    
    def get_labels(self):
        label_list = list(sorted(set(self.flat_labels)))
        label_list.append("[CLS]")
        label_list.append("[SEP]")
        return label_list

    def get_sentences_and_labels(self):
        """
        return : list of sentences : ['I have apple', 'I am here', 'hello ']
        return : list of labels : ['O', 'O', 'B-GPE', ...]
        """
        labels, label, sentences, sentence, flat_labels = [], [], [], [], []
        for line in self.file:
            line = line.strip()
            splits = line.split("\t")
            # if splits[1].startswith('-') or splits[1].startswith(',') or splits[1].startswith('.') or splits[1].startswith(':'):
            #     continue
            if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n" :
                if len(label)>0 and len(sentence)>0:
                    sentences.append(" ".join(sentence))
                    labels.append(label)
                    assert len(sentences) == len(labels)
                    sentence = []
                    label = []
                continue
            # if splits[1].startswith('-') or splits[1].startswith(',') or splits[1].startswith('.') or splits[1].startswith(':'):
            #     continue
            # else:
            sentence.append(splits[0])
            label.append(splits[3])
            flat_labels.append(splits[3])

        if len(label)>0 and len(sentence)>0:
            sentences.append(" ".join(sentence))
            labels.append(label)
            

        return sentences, labels, flat_labels

class IswPreprocessor:
    def __init__(self, filename='data/full-isw-release.tsv'):
        logger.info('------ Preprocssing ISW German corpus ------')
        self.file = open(filename, encoding='utf-8')
        self.sentences, self.labels, self.flat_labels = self.get_sentences_and_labels()

        logger.info("Number of sentences: {0} ".format(len(self.sentences)))
        logger.info("Number of tags: {0} ".format(len(self.get_labels())))

    def get_labels(self):
        label_list = list(map(lambda x: x if x != 'NONE' else 'O', sorted(set(self.flat_labels))))
        label_list.append("[CLS]")
        label_list.append("[SEP]")
        return label_list

    def get_sentences_and_labels(self):
        """
        return : list of sentences : ['I have apple', 'I am here', 'hello ']
        return : list of labels : ['O', 'O', 'B-GPE', ...]
        """
        labels, label, sentences, sentence, flat_labels = [], [], [], [], []
        for line in self.file:
            if line.startswith("idx") or line.startswith("0") or line.startswith("NONE"):
                continue
            line = line.strip()
            splits = line.split("\t")
            if '?' in splits[2] or '.' in splits[2] :
                if len(label)>0 and len(sentence)>0:
                    sentences.append(" ".join(sentence))
                    labels.append(label)
                    sentence = []
                    label = []
                continue
            if splits[3] != 'NONE':
                sentence.append(splits[3])
                label.append(splits[6])
                flat_labels.append(splits[6])

        if len(label)>0 and len(sentence)>0:
            sentences.append(" ".join(sentence))
            labels.append(label)

        labels = [list(map(lambda x: x if x != 'NONE' else 'O', i)) for i in labels]

        return sentences, labels, flat_labels


class TweetPreprocessor:
    def __init__(self, filename='data/merged_headlines_annos.compact.tsv'):
        logger.info('------ Preprocssing Tweets corpus ------')
        self.file = open(filename, encoding='utf-8')
        self.ners_vals=[]

    def get_sentences_and_labels(self):
        """
        return : list of sentences : ['I have apple', 'I am here', 'hello ']
        return : list of labels : ['O', 'O', 'B-GPE', ...]
        """
        labels, label, sentences, sentence, flat_labels = [], [], [], [], []
        for line in self.file:
            if line.startswith("#"):
                continue
            line = line.strip()
            splits = line.split("\t")
            if line.startswith("NONE"):
                if len(label)>0 and len(sentence)>0:
                    sentences.append(" ".join(sentence))
                    labels.append(label)
                    sentence = []
                    label = []
                continue
            sentence.append(splits[1])
            label.append(splits[3])
            flat_labels.append(splits[3])
        
        if len(label)>0 and len(sentence)>0:
            sentences.append(" ".join(sentence))
            labels.append(label)
            
        labels = [list(map(lambda x: x if x != 'NONE' else 'O', i)) for i in labels]
        self.ners_vals = list(map(lambda x: x if x != 'NONE' else 'O', set(flat_labels)))

        logger.info("Number of sentences: {0} ".format(len(sentences)))
        logger.info("Number of tags: {0} ".format(len(self.ners_vals)))

        return sentences, labels

    def get_tag2idx_idx2tag(self):
        """
        return : dict of tag2idx : {'B-ADD': 0, 'B-AGE': 1, 'B-ART': 2, 'B-CARDINAL': 3,'B-CREAT': 4, ...}
        return : dict of idx2tag : inverted
        """
        tag2idx = {t: i for i, t in enumerate(sorted(self.ners_vals), 1)}
        idx2tag = {i: t for t, i in tag2idx.items()}
        return tag2idx, idx2tag
# filename='data/onto.train.ner.sample'
# pre = OntoPreprocessor()
# print('sen', pre.sentences)
# print('label', pre.labels)
# print('lab_list', pre.get_labels())
# # print('data', pre.readfile(filename))

# pre2 = IswPreprocessor()
# print('sen', pre2.sentences[:2])
# print('label', pre2.labels[:2])
# print('lab_list', pre2.get_labels())
