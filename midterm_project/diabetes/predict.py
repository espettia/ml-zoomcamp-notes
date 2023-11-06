import pickle
from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model.bin'

with open(model_file,'rb') as f_in:
    dv,model = pickle.load(f_in)

app = Flask('diabetes')

#Assume it is a simple patient
@app.route('/predict',methods=['POST'])
def predict():
    patient=request.get_json()
    if 'genhlth' in patient:
        del my_dict['genhlth']
    X = dv.transform([patient])
    y_pred = model.predict_proba(X)[0,1]
    diagnosis = y_pred>=0.5
    result = {
        'diabetes_probability': float(y_pred),
        'diabetes': bool(diagnosis)
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
