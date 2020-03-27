from flask import Flask,request,jsonify
from flask_cors import CORS
import json
import os
import logging
import argparse

from predict import Ner

app = Flask(__name__)
CORS(app)

log_dir = "./logs"
model_dir = "./models"
model = Ner(model_dir)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--models_dir",
                        default='./models',
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
args = parser.parse_args()

def setup():
    logs_dir = log_dir
    models_dir = args.models_dir

    if not os.path.exists(logs_dir):
        logger.info('Dir {} doesnot exist, creating it ...'.format(logs_dir))
        os.makedirs(logs_dir)

    if not os.path.exists(models_dir):
        logger.info('Dir ${} doesnot exist, creating it ...'.format(models_dir))
        os.makedirs(models_dir)

    logging.basicConfig(filename='{}/ner.log'.format(logs_dir), level=logging.INFO, filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@app.route("/predict",methods=['POST'])
def predict():
    request_data = request.get_json()
    text = request_data['sentence']
    logger.info('Receive sentence for predicting')
    try:
        out = model.predict(text)
        return jsonify({"result":out})
    except Exception as e:
        print(e)
        return jsonify({"result":"Model Failed"})

@app.route("/info",methods=['GET'])
def info():
    model_config = json.load(open("{0}/model_config.json".format(args.models_dir)))
    logger.info('Load model config')
    return jsonify({"config:" : model_config})

if __name__ == "__main__":
    setup()
    app.run('0.0.0.0',port=8080)