import pickle

with open('model1.bin',mode='rb') as fmodel, open('dv.bin',mode='rb') as fdv:
    dv=pickle.load(fdv)
    model=pickle.load(fmodel)

client={"job": "retired", "duration": 445, "poutcome": "success"}

X = dv.transform(client)
print(model.predict_proba(X)[:,1])
