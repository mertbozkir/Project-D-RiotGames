# importing libraries

import os
import yaml
import sys
import pandas as pd
import ast
import numpy as np
from sklearn.model_selection import train_test_split
min = yaml.safe_load(open('params.yaml'))['preprocess']['min']

file_name_10 = sys.argv[1]
file_name_15 = sys.argv[2]


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





def data_preprocess(minute = 15):
    if minute == 10:
        file = file_name_10
    else:
        file = file_name_15
    dataframe = pd.read_csv(file)
    dataframe.drop(['gameId', 'redWins'], axis = 1, inplace = True)
    dataframe.rename(columns = {'blueDragnoType':'blueDragonType', 'redDragnoType':'redDragonType'}, inplace = True)
    str_to_value(dataframe, 'blueFirstTowerLane')
    str_to_value(dataframe, 'redFirstTowerLane')
    str_to_value(dataframe, 'blueDragonType', cat = False)
    str_to_value(dataframe, 'redDragonType', cat = False)
    
    return dataframe



dataframe = data_preprocess(minute = min)


# create folder to save file
data_path = 'prepared/'
os.makedirs(data_path, exist_ok=True)

# saving prepared data
#train.to_csv(os.path.join(data_path, "out_train.csv"))

test_size_yaml  = yaml.safe_load(open('params.yaml'))['preprocess']['test_size']


X_train, X_test, y_train, y_test = train_test_split(dataframe.drop(['blueWins'], axis = 1), dataframe['blueWins'], test_size=test_size_yaml, random_state=42)

X_train['blueWins'] = y_train
X_test['blueWins'] = y_test

X_train.to_csv(os.path.join(data_path, 'train_dataframe.csv'))
X_test.to_csv(os.path.join(data_path, 'test_dataframe.csv'))
