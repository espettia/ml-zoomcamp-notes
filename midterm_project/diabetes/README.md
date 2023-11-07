## Datatalks - Midterm Project
### Diabetes diagnosis prediction
#Problem Description
The dataset is located in UCIrvine Repository at https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators .

According to the website:
The Diabetes Health Indicators Dataset contains healthcare statistics and lifestyle survey information about people in general along with their diagnosis of diabetes. The 35 features consist of some demographics, lab test results, and answers to survey questions for each patient. The target variable for classification is whether a patient has diabetes, is pre-diabetic, or healthy.
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
