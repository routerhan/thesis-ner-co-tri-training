import pandas as pd


class Preprocessor:
    def __init__(self, filename):
        self.row_isw_data = self.load_isw_tsv_file(filename)
        self.cleaned_isw_data = self.clean_isw_data()

    def load_isw_tsv_file(self, filename='data/test-full-isw-release.tsv'):
        isw_data = pd.read_csv(filename, quotechar='"',
                               delimiter="\t", skiprows=None)
        print("Total number of rows", len(isw_data))
        print("Total number of sentences", len(isw_data.fileid.unique()))
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
        return isw_drop_non

    def get_list_of_sentences(self):
        """
        return : list of sentences : ['I have apple', 'I am here', 'hello ']
        """
        data = self.cleaned_isw_data
        # Group the sentence with its fileid
        agg_func = lambda s: [(token, lem, ner) for token, lem, ner in zip(s["token"].values.tolist(),
                                                    s["lemma"].values.tolist(),
                                                    s["ontoNer"].values.tolist())]
        grouped = data.groupby("fileid").apply(agg_func)
        grouped_all = [s for s in grouped]

        sentences = [" ".join([s[0] for s in sent]) for sent in grouped_all]
        return sentences

    def get_list_of_nerlabels(self):
        """
        return : list of labels : ['O', 'O', 'B-GPE', ...]
        """
        data = self.cleaned_isw_data
        # Group the sentence with its fileid
        agg_func = lambda s: [(token, lem, ner) for token, lem, ner in zip(s["token"].values.tolist(),
                                                    s["lemma"].values.tolist(),
                                                    s["ontoNer"].values.tolist())]
        grouped = data.groupby("fileid").apply(agg_func)
        grouped_all = [s for s in grouped]

        labels = [[s[2] for s in label] for label in grouped_all]

        return labels

    def get_tag2idx(self):
        """
        return : dict of ner label with idx : {'B-ADD': 0, 'B-AGE': 1, 'B-ART': 2, 'B-CARDINAL': 3,'B-CREAT': 4, ...}
        """
        data = self.cleaned_isw_data
        # ners_vals : list of ner labels
        ners_vals = list(set(data["ontoNer"].values))
        # Set as dict {key:idx}
        tag2idx = {t: i for i, t in enumerate(sorted(ners_vals))}
        return tag2idx





# filename = 'data/test-full-isw-release.tsv'
# pre = Preprocessor(filename)

# sentences = pre.get_list_of_sentences()
# labels = pre.get_list_of_nerlabels()

# print(labels[0])
