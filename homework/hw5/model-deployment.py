from flask import Flask, jsonify, request
import pickle

with open('model1.bin',mode='rb') as fmodel, open('dv.bin',mode='rb') as fdv:
    dv=pickle.load(fdv)
    model=pickle.load(fmodel)

def predict(customer, dv, model):
    X = dv.transform(customer)
    y_pred = model.predict_proba(X)[:,1]
    return y_pred

app = Flask('churn')

@app.route('/predict',methods=['POST'])
def predict_service():
    customer = request.get_json()
    pred = predict(customer,dv,model)
    churn = pred>=.5
    result = {
            'churn_probability':float(pred),
            'churn':bool(churn)
    }
    return jsonify(result)

if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0',port=9696)


