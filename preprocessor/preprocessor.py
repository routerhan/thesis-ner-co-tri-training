import pandas as pd


class IswPreprocessor:
    def __init__(self, filename):
        print(' ------ Preprocssing ISW German corpus ------')
        self.file = self.load_isw_tsv_file(filename)
        self.ners_vals=[]

    def load_isw_tsv_file(self, filename='data/test-full-isw-release.tsv'):
        file = open(filename, encoding='utf-8')
        return file

    def get_list_of_sentences_labels(self):
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
        self.ners_vals = list(map(lambda x: x if x != 'NONE' else 'O', set(flat_labels)))
        
        print("number of sentences:", len(sentences))
        print('num of tags :', len(self.ners_vals))

        return sentences, labels

    def get_tag2idx_idx2tag(self):
        """
        return : dict of tag2idx : {'B-ADD': 0, 'B-AGE': 1, 'B-ART': 2, 'B-CARDINAL': 3,'B-CREAT': 4, ...}
        return : dict of idx2tag : inverted
        """
        tag2idx = {t: i for i, t in enumerate(sorted(self.ners_vals))}
        idx2tag = {i: t for t, i in tag2idx.items()}
        return tag2idx, idx2tag


class TweetPreprocessor:
    def __init__(self, filename='data/merged_headlines_annos.compact.tsv'):
        print(' ------ Preprocssing Tweets corpus ------')
        self.file = open(filename, encoding='utf-8')
        self.ners_vals=[]

    def get_list_of_sentences_labels(self):
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
        print("Total number of tweets", len(sentences))
        print("Total number of ner tags in tweets", len(self.ners_vals))

        return sentences, labels

    def get_tag2idx_idx2tag(self):
        """
        return : dict of tag2idx : {'B-ADD': 0, 'B-AGE': 1, 'B-ART': 2, 'B-CARDINAL': 3,'B-CREAT': 4, ...}
        return : dict of idx2tag : inverted
        """
        tag2idx = {t: i for i, t in enumerate(sorted(self.ners_vals))}
        idx2tag = {i: t for t, i in tag2idx.items()}
        return tag2idx, idx2tag