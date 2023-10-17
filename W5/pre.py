
#PREDICT SCRIPT
import pickle
from flask import Flask, request, jsonify
filename='model.bin'
with open(filename,mode='rb') as f:
    dv, model = pickle.load(f)

app = Flask('predict')

@app.route('/predict',methods=['POST'])
def predict():
    customer = request.get_json()
    X = dv.transform([customer])
    y_pred= model.predict_proba(X)[:,1]
    churn = bool(y_pred>=0.5)

    result={
        'churn_prediction': float(y_pred),
        'churn':churn
    }
    return jsonify(result)
    
if __name__=='__main__':
    app.run(host='0.0.0.0',port='9696',debug=True)
