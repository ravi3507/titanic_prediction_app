import numpy as np
import pandas as pd

import streamlit as st
st.set_page_config(page_title='Titanic Survival Prediction App',
    layout='wide')
import csv

st.title("App to Predict Survival Chances in Titanic")


def mode(lum):
    train = pd.read_csv('C:/Users/rocki/OneDrive/Desktop/mlapp/train.csv')
    test = pd.read_csv('C:/Users/rocki/OneDrive/Desktop/mlapp/test.csv')
    train.describe(include="all")
    train["Age"] = train["Age"].fillna(-0.5)
    test["Age"] = test["Age"].fillna(-0.5)
    bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
    labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
    test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)
    train["CabinBool"] = (train["Cabin"].notnull().astype('int'))
    test["CabinBool"] = (test["Cabin"].notnull().astype('int'))
    train = train.drop(['Cabin'], axis = 1)
    test = test.drop(['Cabin'], axis = 1)
    train = train.drop(['Ticket'], axis = 1)
    test = test.drop(['Ticket'], axis = 1)
    train = train.fillna({"Embarked": "S"})
    combine = [train, test]
    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    for dataset in combine:
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
        'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
        
        dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)
    mr_age = train[train["Title"] == 1]["AgeGroup"].mode() #Young Adult
    miss_age = train[train["Title"] == 2]["AgeGroup"].mode() #Student
    mrs_age = train[train["Title"] == 3]["AgeGroup"].mode() #Adult
    master_age = train[train["Title"] == 4]["AgeGroup"].mode() #Baby
    royal_age = train[train["Title"] == 5]["AgeGroup"].mode() #Adult
    rare_age = train[train["Title"] == 6]["AgeGroup"].mode() #Adult

    age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}
    for x in range(len(train["AgeGroup"])):
        if train["AgeGroup"][x] == "Unknown":
            train["AgeGroup"][x] = age_title_mapping[train["Title"][x]]
            
    for x in range(len(test["AgeGroup"])):
        if test["AgeGroup"][x] == "Unknown":
            test["AgeGroup"][x] = age_title_mapping[test["Title"][x]]
    age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
    train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
    test['AgeGroup'] = test['AgeGroup'].map(age_mapping)

    train.head()

    train = train.drop(['Age'], axis = 1)
    test = test.drop(['Age'], axis = 1)
    train = train.drop(['Name'], axis = 1)
    test = test.drop(['Name'], axis = 1)
    sex_mapping = {"male": 0, "female": 1}
    train['Sex'] = train['Sex'].map(sex_mapping)
    test['Sex'] = test['Sex'].map(sex_mapping)
    embarked_mapping = {"S": 1, "C": 2, "Q": 3}
    train['Embarked'] = train['Embarked'].map(embarked_mapping)
    test['Embarked'] = test['Embarked'].map(embarked_mapping)
    for x in range(len(test["Fare"])):
        if pd.isnull(test["Fare"][x]):
            pclass = test["Pclass"][x] #Pclass = 3
            test["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)
            
    #map Fare values into groups of numerical values
    train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])
    test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])

    #drop Fare values
    train = train.drop(['Fare'], axis = 1)
    test = test.drop(['Fare'], axis = 1)
    from sklearn.model_selection import train_test_split

    predictors = train.drop(['Survived', 'PassengerId'], axis=1)
    target = train["Survived"]
    x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)

    from sklearn.ensemble import GradientBoostingClassifier

    gbk = GradientBoostingClassifier()
    gbk.fit(x_train, y_train)
    x=pd.DataFrame(lum).T
    x.columns=['Pclass','Sex','SibSp','Parch','Embarked','AgeGroup','CabinBool','Title','FareBand']
    y=gbk.predict(x)
    return y
genre=st.radio(
    "Ticket Class",
    ('1st', '2nd', '3rd'))
if genre == '1st':
    a=1
elif genre=='2nd':
    a=2
else:
    a=3
sex=st.radio("Gender",("Male","Female",))
if sex=="Male":
    b=0
else:
    b=1
title=st.radio("Title",("Mr", "Miss", "Mrs", "Master", "Royal", "Rare"))
if title=="Mr":
    h=1
elif title=="Master":
    h=4
elif title=="Miss":
        h=2
elif title=="Mrs":
    h=3
elif title=="Royal":
    h=5
else:
    h=6
embarked=st.radio("Port of Embarkation",("Cherbourg", "Queenstown", "Southampton"))
if embarked=="Southampton":
    e=1 
elif embarked=="Cherbourg":
    e=2
else:
    e=3
age=st.radio("AgeGroup",('Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior'))
if age=="Baby":
    f=1
elif age=="Child":
    f=2
elif age=="Teenager":
    f=3
elif age=="Student":
    f=4
elif age=="Young Adult":
    f=5
elif age=="Adult":
    f=6
else:
    f=7
cabin=st.radio("Cabin Number",("First","Second"))
if cabin=="First":
    g=0
else:
    g=1
parch=st.radio("No. of parents / children aboard the Titanic",("None", 1, 2, 3, 4,5, 6))
if parch=="None":
    d=0
else:
    d=parch
sib=st.radio("	No. of siblings / spouses aboard the Titanic",("None", 1,2,3, 4,5))
if sib=="None":
    c=0
else:
    c=sib
if genre=="1st":
    i=1
elif genre=="2nd":
    i=2
else:
    i=3

values=[a,b,c,d,e,f,g,h,i]
click=st.button("Click to See Survival Chances")
if click:
    if mode(values)==0:
        st.write("**Less Survival chances**")
    else:
        st.write("High Survival chances")
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://www.thespruceeats.com/thmb/ytOWw19bNbrd7iT0T-xrPISR9ro=/940x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/GettyImages-738790035-5c565bfdc9e77c000102c641.jpg");
             background-attachment:scroll;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()






