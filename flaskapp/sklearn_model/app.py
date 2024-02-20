#!/usr/bin/env python3

from flask import Flask, request,jsonify

import joblib
import pandas as pd
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
pipeline = pickle.load(open('pipeline.pkl', 'rb'))

def islist(obj):
  return True if ("list" in str(type(obj))) else False

@app.route('/predict', methods=['POST'])
def update():
    # record = json.loads(request.data)
    json_ = request.json
    print(json_)
    if islist(json_['PassengerId']):
      entry = pd.DataFrame(json_)
    else:
      entry = pd.DataFrame([json_])
    entry_transformed = pipeline.transform(entry)
    prediction = model.predict(entry_transformed)
    res = {'predictions': {}}
    for i in range(len(prediction)):
      res['predictions'][i + 1] = int(prediction[i])
    return res, 200 # {'prediction': int(prediction[0])}

if __name__ == "__main__":
  app.run(debug = True)
