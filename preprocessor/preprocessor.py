import pandas as pd


class IswPreprocessor:
    def __init__(self, filename):
        print(' ------ Preprocssing ISW German corpus ------')
        self.row_isw_data = self.load_isw_tsv_file(filename)
        self.cleaned_isw_data = self.clean_isw_data()

    def load_isw_tsv_file(self, filename='data/test-full-isw-release.tsv'):
        isw_data = pd.read_csv(filename, quotechar='"',
                               delimiter="\t", skiprows=None)
        return isw_data

    def clean_isw_data(self, selected_cols=[]):
        """
        :return: clean isw_data
        """
        # Keep only selected cols
        selected_cols = ['fileid', 'token', 'lemma', 'ontoNer']
        isw_set = self.row_isw_data[selected_cols]

        # Clean up incorrect rows  e.g. fileid -> total 82 of it
        isw_set = isw_set[isw_set.fileid != "fileid"]

        # Drop empty token
        isw_drop_non = isw_set[isw_set.lemma != "NONE"]
        isw_drop_non.reset_index(drop=True, inplace=True)

        # Replace NONE tag with "O"
        isw_drop_non['ontoNer'].replace(
            to_replace='NONE', value='O', inplace=True)

        print("Total number of sentences", len(isw_drop_non.fileid.unique()))
        print("Total number of ner tags in isw", len(list(set(isw_drop_non["ontoNer"].values))))
        return isw_drop_non

    def get_list_of_sentences_labels(self):
        """
        return : list of sentences : ['I have apple', 'I am here', 'hello ']
        return : list of labels : ['O', 'O', 'B-GPE', ...]
        """
        data = self.cleaned_isw_data
        # Group the sentence with its fileid
        agg_func = lambda s: [(token, lem, ner) for token, lem, ner in zip(s["token"].values.tolist(),
                                                    s["lemma"].values.tolist(),
                                                    s["ontoNer"].values.tolist())]
        grouped = data.groupby("fileid").apply(agg_func)
        grouped_all = [s for s in grouped]

        sentences = [" ".join([s[0] for s in sent]) for sent in grouped_all]
        labels = [[s[2] for s in label] for label in grouped_all]
        return sentences, labels

    def get_tag2idx_idx2tag(self):
        """
        return : dict of ner label with idx : {'B-ADD': 0, 'B-AGE': 1, 'B-ART': 2, 'B-CARDINAL': 3,'B-CREAT': 4, ...}
        """
        data = self.cleaned_isw_data
        # ners_vals : list of ner labels
        ners_vals = list(set(data["ontoNer"].values))
        # Set as dict {key:idx}
        tag2idx = {t: i for i, t in enumerate(sorted(ners_vals))}
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




# filename = 'data/test-full-isw-release.tsv'
# pre = IswPreprocessor(filename)

# sentences = pre.get_list_of_sentences()
# labels = pre.get_list_of_nerlabels()

# print(labels[0])
