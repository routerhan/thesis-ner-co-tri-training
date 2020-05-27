import os
import joblib
import xml.etree.ElementTree as ET
import argparse
import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def list_files(dir):
    r = []
    subdirs = [x[0] for x in os.walk(dir)]
    for subdir in subdirs:
        files = os.walk(subdir).__next__()[2]
        if (len(files) > 0):
            for file in files:
                if file.endswith('.xml'):
                    r.append(subdir + "/" + file)
    return r

def get_sentences(xml_list):
    all_sentences = []
    for i, xml in enumerate(xml_list):
        tree = ET.parse(xml)
        words= tree.findall('.//w')
        sentences, sentence = [], []
        for w in words:
            if '?' in w.text or '.' in w.text or '!' in w.text:
                if len(sentence)>10:
                    sentences.append(" ".join(sentence))
                    sentence = []
                continue
            if w.text != '"':
                sentence.append(w.text)
        if len(sentence)>10:
            sentences.append(" ".join(sentence))
        all_sentences = all_sentences + sentences
    return all_sentences

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year_data_dir",
                        default='2017',
                        type=str,
                        required=True,
                        help="the folder of movie subtiles for preprocessing.")
    args = parser.parse_args()

    directory = os.path.join('/home/chinchen/thesis/OpenSubtitles/xml/de/', args.year_data_dir)
    xml_list = list_files(directory)
    all_sentences = get_sentences(xml_list)
    logger.info("Year: {}, Num of sentences: {}".format(args.year_data_dir, len(all_sentences)))
    # Save to sentences
    joblib.dump(all_sentences,'unlabel_sentences/{}_sentences.pkl'.format(args.year_data_dir))
    logger.info("Save unlabel sentences as pickle file: unlabel_sentences/{}".format(args.year_data_dir))

    # Save to machine translation dir as txt file
    s = joblib.load('unlabel_sentences/{}_sentences.pkl'.format(args.year_data_dir))
    with open("machine_translation/{}_de_sents.txt".format(args.year_data_dir), "w", encoding="utf-8") as writer:
        for sent in s:
            writer.write(str(sent)+'\n')
    writer.close()
    logger.info("Save unlabel sentences as txt file: machine_translation/{}".format(args.year_data_dir))



if __name__ == '__main__':
    if not os.path.exists('unlabel_sentences'):
        os.makedirs('unlabel_sentences')
    main()
