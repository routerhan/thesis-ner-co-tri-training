from flask import Flask,request,jsonify
from flask_cors import CORS
import json
from predict import Ner

app = Flask(__name__)
CORS(app)

model_dir = "./models"
model = Ner(model_dir)

@app.route("/predict",methods=['POST'])
def predict():
    request_data = request.get_json()
    text = request_data['sentence']
    try:
        out = model.predict(text)
        return jsonify({"result":out})
    except Exception as e:
        print(e)
        return jsonify({"result":"Model Failed"})

@app.route("/info",methods=['GET'])
def info():
    model_config = json.load(open("./models/model_config.json"))
    return jsonify({"config:" : model_config})

if __name__ == "__main__":
    app.run('0.0.0.0',port=8080)