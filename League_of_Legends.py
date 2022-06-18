import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import ast
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate, GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

st.set_option('deprecation.showPyplotGlobalUse', False)

# from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier


import warnings
pd.set_option('display.max_columns', None)
warnings.simplefilter(action = 'ignore', category = Warning)



def str_to_value(dataframe, col, cat = True):
    temp_col = dataframe[col].apply(lambda x: ast.literal_eval(x))
    if cat:
        temp_col = temp_col.apply(lambda x: x[0] if len(x) != 0 else np.nan)
        
        if col.startswith('blue') == True:
            dataframe['blue_top_lane_first_tower'] = dataframe[col].apply(lambda x: 1 if x == 'TOP_LANE' else 0)  
            dataframe['blue_mid_lane_first_tower'] = dataframe[col].apply(lambda x: 1 if x == 'MID_LANE' else 0)
            dataframe['blue_bot_lane_first_tower'] = dataframe[col].apply(lambda x: 1 if x == 'BOT_LANE' else 0)
        else: 
            dataframe['red_top_lane_first_tower'] = dataframe[col].apply(lambda x: 1 if x == 'TOP_LANE' else 0)  
            dataframe['red_mid_lane_first_tower'] = dataframe[col].apply(lambda x: 1 if x == 'MID_LANE' else 0)
            dataframe['red_bot_lane_first_tower'] = dataframe[col].apply(lambda x: 1 if x == 'BOT_LANE' else 0)

        dataframe.drop(col, axis = 1, inplace = True) 
    else: 
        if col.startswith('blue') == True:
            for i in ['earth', 'air', 'fire', 'water', 'elder']:
                dataframe['blue_' + i + '_dragon'] = dataframe[col].apply(lambda x: x.count(i.upper() + '_DRAGON'))
            dataframe.drop([col], axis = 1, inplace = True)
    
        else:
            for i in ['earth', 'air', 'fire', 'water', 'elder']:
                dataframe['red_' + i + '_dragon'] = dataframe[col].apply(lambda x: x.count(i.upper() + '_DRAGON'))
            dataframe.drop([col], axis = 1, inplace = True)

def feature_engineering(df):
    df['blue-red_total_golds_diff'] = df['blueTotalGolds'] - df['redTotalGolds']
    df['blue-red_current_golds_diff'] = df['blueCurrentGolds'] - df['redCurrentGolds']
    df['blue-red_total_level_diff'] = df['blueTotalLevel'] - df['redTotalLevel']
    df['blue-red_avg_level_diff'] = df['blueAvgLevel'] - df['redAvgLevel']
    df['blue-red_total_minion_diff'] = df['blueTotalMinionKills'] - df['redTotalMinionKills']
    df['blue-red_jungle_minion_diff'] = df['blueTotalJungleMinionKills'] - df['redTotalJungleMinionKills']
    df['blue-red_kill_diff'] = df['blueKill'] - df['redKill']
    df['blue-red_death_diff'] = df['blueDeath'] - df['redDeath']
    df['blue-red_assist_diff'] = df['blueAssist'] - df['redAssist']
    df['blue_total_wards'] = df['blueWardPlaced'] - df['redWardKills']
    df['red_total_wards'] = df['redWardPlaced'] - df['blueWardKills']
    df['blue-red_tower_diff'] = df['blueTowerKills'] - df['redTowerKills']
    df['blue-red_dragon_diff'] = df['blueDragon'] -df['redDragon']
    df['red_monster_kills'] = df['redDragon'] + df['redRiftHeralds']
    df['blue_monster_kills'] = df['blueDragon'] + df['blueRiftHeralds']



def data_preprocess(path = "League Of Legends Challenger Rank Game-10min,15min/", min = 15):
    file = path + f'Challenger_Ranked_Games_{min}minute.csv'
    dataframe = pd.read_csv(file)
    dataframe.drop(['gameId', 'redWins'], axis = 1, inplace = True)
    dataframe.rename(columns = {'blueDragnoType':'blueDragonType', 'redDragnoType':'redDragonType'}, inplace = True)
    str_to_value(dataframe, 'blueFirstTowerLane')
    str_to_value(dataframe, 'redFirstTowerLane')
    str_to_value(dataframe, 'blueDragonType', cat = False)
    str_to_value(dataframe, 'redDragonType', cat = False)
    feature_engineering(dataframe)
    return dataframe

def plot_importance(model, features, num = 10, save = False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize = (10, 10))
    sns.set(font_scale = 1)
    sns.barplot(x = 'Value', y = 'Feature', 
                data = feature_imp.sort_values(by = 'Value', ascending = False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


def model_development(dataframe, model = 'LightGBM', cv = True, cv_number = 10, grid_cv = False):
    lgbm = dt = logreg = rf = None
    X = dataframe.drop('blueWins', axis = 1)
    y = dataframe['blueWins']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    if model == 'LightGBM':
        lgbm = LGBMClassifier()
        lgbm.fit(X_train, y_train)
        y_pred = lgbm.predict(X_test)
        acc = accuracy_score(y_pred, y_test)
        if cv:
            cv_results = cross_validate(lgbm, X, y, cv = cv_number)

    if model == 'Random_Forest':
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_pred, y_test)
        if cv:
            cv_results = cross_validate(rf, X, y, cv = cv_number)
        if grid_cv:
            rf_params = {'max_depth': [5, 8, None],
            'max_features': [3, 5, 7, 'auto'],
            'min_samples_split': [2, 5, 8, 15, 20],
            'n_estimators': [100, 200, 500]}
            rf_best_grid = GridSearchCV(rf, rf_params, cv = 5, n_jobs = -1, verbose = True).fit(X, y)
            rf = rf.set_params(**rf_best_grid.best_params_, random_state = 17).fit(X, y)

            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            acc = accuracy_score(y_pred, y_test)          

    if model == 'Logistic_Regression':
        logreg = LogisticRegression()

        logreg.fit(X_train, y_train)

        y_pred = logreg.predict(X_test)

        acc = accuracy_score(y_pred, y_test)



    if model == 'Decision_Tree':
        dt = DecisionTreeClassifier(max_depth=6, random_state=1)

        dt.fit(X_train, y_train)

        y_pred = dt.predict(X_test)

        acc = accuracy_score(y_pred, y_test)

        if cv:
            cv_results = cross_validate(dt, X, y, cv = cv_number)
    
    label = f'{model} accuracy score'
    value = f'{acc:0.4f}'

    st.metric(label, value, delta=None, delta_color="normal")


    if cv:
        st.write(f"(Cross Validation Score): {round(cv_results['test_score'].mean(), 4)}")
    
    st.json(classification_report(y_test, y_pred, output_dict = True))
    st.write('Confusion Matrix')
    st.write(confusion_matrix(y_test, y_pred))

    model_dict = {'LightGBM': lgbm, 'Decision_Tree': dt, 'Logistic_Regression': logreg, 'Random_Forest': rf}

    return X, model_dict[model]
    
    
    # st.pyplot(plot_importance(model_dict[model], X, num = 10))



