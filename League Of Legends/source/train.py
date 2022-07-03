import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import sys
import yaml
import pickle
import os
import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression







def model_development(train_features, model_name = "LGBMClassifier"):

    X_train = train_features.drop('blueWins', axis = 1)
    y_train = train_features['blueWins']

    if model_name == 'LGBMClassifier':
        model = LGBMClassifier()
        
        model.fit(X_train, y_train)


    if model_name == 'RandomForestClassifier':
        model = RandomForestClassifier()
        
        model.fit(X_train, y_train)
        

    if model_name == 'LogisticRegression':
        model = LogisticRegression(solver='liblinear', max_iter=1)

        model.fit(X_train, y_train)


    if model_name == 'DecisionTreeClassifier':
        model = DecisionTreeClassifier(max_depth=6, random_state=1)

        model.fit(X_train, y_train)




    with open(f'models/{model_name}.pkl', 'wb') as model_output:
        pickle.dump(model, model_output)




    



train_path = sys.argv[1]
train_features = pd.read_csv(train_path, index_col=0)
train_features = train_features.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))


model_name_yaml = yaml.safe_load(open('params.yaml'))['train']['model']
model_development(train_features, model_name = model_name_yaml)
