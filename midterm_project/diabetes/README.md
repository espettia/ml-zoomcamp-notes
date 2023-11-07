## Datatalks - Midterm Project
### Diabetes diagnosis prediction
#Problem Description
The dataset is located in UCIrvine Repository at https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators .

According to the website:
The Diabetes Health Indicators Dataset contains healthcare statistics and lifestyle survey information about people in general along with their diagnosis of diabetes. The 35 features consist of some demographics, lab test results, and answers to survey questions for each patient. The target variable for classification is whether a patient has diabetes, is pre-diabetic, or healthy.\
\
The dataset consists on the following features:
| Variable Name |	Demographic	Description |
| ------------- | ----------------------- |
|Diabetes_binary	|	0 = no diabetes 1 = prediabetes or diabetes	|
|HighBP	|		0 = no high BP 1 = high BP	|
|HighChol	|	0 = no high cholesterol 1 = high cholesterol	|
|CholCheck		|	0 = no cholesterol check in 5 years 1 = yes cholesterol check in 5 years|
|BMI		|	Body Mass Index		|
|Smoker	|		Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes] 0 = no 1 = yes		no|
|Stroke	|	(Ever told) you had a stroke. 0 = no 1 = yes		no|
|HeartDiseaseorAttack	|	coronary heart disease (CHD) or myocardial infarction (MI) 0 = no 1 = yes		no|
|PhysActivity		|	physical activity in past 30 days - not including job 0 = no 1 = yes		no|
|Fruits	|	Consume Fruit 1 or more times per day 0 = no 1 = yes		no|
|Veggies		|	Consume Vegetables 1 or more times per day 0 = no 1 = yes		no|
|HvyAlcoholConsump	|		Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week) 0 = no 1 = yes		no|
|AnyHealthcare	|		Have any kind of health care coverage, including health insurance, prepaid plans such as HMO, etc. 0 = no 1 = yes		no|
|NoDocbcCost	|		Was there a time in the past 12 months when you needed to see a doctor but could not because of cost? 0 = no 1 = yes		no|
|GenHlth	|	Would you say that in general your health is: scale 1-5 1 = excellent 2 = very good 3 = good 4 = fair 5 = poor		no|
|MentHlth	|		Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good? scale 1-30 days		no|
|PhysHlth	|		Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? scale 1-30 days		no|
|DiffWalk	|		Do you have serious difficulty walking or climbing stairs? 0 = no 1 = yes		no|
|Sex	| Sex	0 = female 1 = male		no|
|Age	|	Age	13-level age category (_AGEG5YR see codebook) 1 = 18-24 9 = 60-64 13 = 80 or older		no|
|Education	| Education Level	Education level (EDUCA see codebook) scale 1-6 1 = Never attended school or only kindergarten 2 = Grades 1 through 8 (Elementary) 3 = Grades 9 through 11 (Some high school) 4 = Grade 12 or GED (High school graduate) 5 = College 1 year to 3 years (Some college or technical school) 6 = College 4 years or more (College graduate)		no|
|Income		|	Income scale (INCOME2 see codebook) scale 1-8 1 = less than $10,000 5 = less than $35,000 8 = $75,000 or more		no|
## EDA
Three scores were used to measure feature importance. According to the following results,
| RISK RATIO |
|--------------------------------------|
|                 Feature |         0 |         1 |max |
|   heartdiseaseorattack| 0.859828 | 2.347570 | 2.347570|
|                 stroke | 0.948060 | 2.232434 | 2.232434|
|               diffwalk | 0.757696 | 2.197808  |2.197808|
|                 highbp|  0.434612 | 1.752259  |1.752259|
|               highchol|  0.575075 | 1.576887|  1.576887|
|           physactivity | 1.521725 | 0.831646 | 1.521725|
|                veggies | 1.288062 | 0.932969 | 1.288062|
|            nodocbccost|  0.975105 | 1.269803 | 1.269803|
|                smoker | 0.867821 | 1.166503 | 1.166503 |
|                 fruits | 1.135693 | 0.921705 | 1.135693 |
|                   sex | 0.931266|  1.087497 | 1.087497 |
|     hvyalcoholconsump | 1.035004 | 0.419417 | 1.035004 |
|             cholcheck | 0.179429 | 1.031916 | 1.031916 |
|         anyhealthcare | 0.833940 | 1.008559 | 1.008559 |
\
We observe that having a heart disease or attack is a good indicator of diabetes, along with ever having a stroke and having difficulty walking. Males are slightly more likely to have diabetes than females.
Interestingly, heavy alcohol consumption is slightly related to not having diabetes, however other effects on health are wellknown.
| MUTUAL INFORMATION SCORE |
|--------------------------|
|                 Feature |   score|
|highbp                 | 0.034604|
|age                   |  0.019799|
|highchol               | 0.019593|
|diffwalk                |0.019550|
|income                 | 0.013187|
|heartdiseaseorattack   | 0.012123|
|education              | 0.007462|
|physactivity           | 0.006539|
|stroke                 | 0.004089|
|cholcheck              | 0.003007|
|hvyalcoholconsump      | 0.002004|
|smoker                 | 0.001760|
|veggies                | 0.001471|
|fruits                 | 0.000843|
|nodocbccost            | 0.000508|
|sex                    | 0.000482|
|anyhealthcare          | 0.000120|
\
According to the mutual information score, the most powerful indicators are general health, high blood presure, age, high cholesterol and difficulty walking.
## Model training 
Three models were trained and compared to find the most accurate according to the ROC AUC score. The following are the results of the models trained on the complete dataset:
| Model | ROC AUC score | Parameters |
|-------|---------------|------------|
| RandomForest | 0.815228 | n_estimators=125,max_depth=10,min_samples_leaf=3, |
|	LogisticRegression |	0.805113 | C=10 |
|	DecisionTree |	0.796986 | max_depth=6,min_samples_leaf=500 |
## Exporting notebook to script	
The script that downloads the data and trains the model is the file 'train.py'. It uses pickle to save the model as a binary.
## Reproducibility	
The notebooks and python scripts contain the necessary code to train the model, execute the web service and test a request. To make use of them install the required dependencies.
## Dependency and enviroment management	
To run the code in your machine, first make sure you have pipenv in your machine:\
pip install pipenv \
Then your working directory has to be the folder containing the files of the project and install the dependencies specified in the Pipfile with \
pipenv install
## Containerization
To build and run the image containing the Flask application and the deployment service in gunicorn, run  the following commands in the directory that contains the 'Dockerfile' file. \
docker build -t diabetes-prediction-project-ml-zoomcamp .
docker run -it --rm -p 9696:9696 diabetes-prediction-project-ml-zoomcamp


