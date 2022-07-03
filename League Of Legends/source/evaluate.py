from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve
import sklearn.metrics as metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import sys
import yaml
import pickle
import re



fold_number = yaml.safe_load(open('params.yaml'))['evaluate']['fold_number']
model_path = sys.argv[1]
train_path = sys.argv[2]
test_path = sys.argv[3]


model_name = model_path[7:-4]
with open(model_path, 'rb') as fd:
    model = pickle.load(fd)





# K-Fold Cross-Validation   
def cross_validation(model, _X, _y, _cv=5):
      '''Function to perform K Folds Cross-Validation
       Parameters
       ----------
      model: Python Class, default=None
              This is the machine learning algorithm to be used for training.
      _X: array
           This is the matrix of features.
      _y: array
           This is the target variable.
      _cv: int, default=5
          Determines the number of folds for cross-validation.
       Returns
       -------
       The function returns a dictionary containing the metrics 'accuracy', 'precision',
       'recall', 'f1' for both training set and validation set.
      '''
      _scoring = ['accuracy', 'precision', 'recall', 'f1']
      results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
      
      return {"Training Accuracy scores": results['train_accuracy'],
              "Mean Training Accuracy": results['train_accuracy'].mean()*100,
              "Training Precision scores": results['train_precision'],
              "Mean Training Precision": results['train_precision'].mean(),
              "Training Recall scores": results['train_recall'],
              "Mean Training Recall": results['train_recall'].mean(),
              "Training F1 scores": results['train_f1'],
              "Mean Training F1 Score": results['train_f1'].mean(),
              "Validation Accuracy scores": results['test_accuracy'],
              "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
              "Validation Precision scores": results['test_precision'],
              "Mean Validation Precision": results['test_precision'].mean(),
              "Validation Recall scores": results['test_recall'],
              "Mean Validation Recall": results['test_recall'].mean(),
              "Validation F1 scores": results['test_f1'],
              "Mean Validation F1 Score": results['test_f1'].mean()
              }







# Grouped Bar Chart for both training and validation data
def plot_result(x_label, y_label, plot_title, train_data, val_data, fold_number):
        '''Function to plot a grouped bar chart showing the training and validation
          results of the ML model in each fold after applying K-fold cross-validation.
         Parameters
         ----------
         x_label: str, 
            Name of the algorithm used for training e.g 'Decision Tree'
          
         y_label: str, 
            Name of metric being visualized e.g 'Accuracy'
         plot_title: str, 
            This is the title of the plot e.g 'Accuracy Plot'
         
         train_result: list, array
            This is the list containing either training precision, accuracy, or f1 score.
        
         val_result: list, array
            This is the list containing either validation precision, accuracy, or f1 score.
         Returns
         -------
         The function returns a Grouped Barchart showing the training and validation result
         in each fold.
        '''
        
        # Set size of plot
        plt.figure(figsize=(12,6))
        labels = [f"{i+1} Fold" for i in range(fold_number)]
        # labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold", "6th Fold", "7th Fold", "8th Fold", "9th Fold", "10th Fold"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.40000, 1)
        plt.bar(X_axis-0.2, train_data, 0.4, color='blue', label='Training')
        plt.bar(X_axis+0.2, val_data, 0.4, color='red', label='Validation')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig(f'../plots/cross_validation.png')


 









train = pd.read_csv(train_path, index_col=0)
train = train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

test = pd.read_csv(test_path, index_col=0)
test = test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

X = train.append(test)
y = X['blueWins']
X = X.drop('blueWins', axis = 1)

cross_val_results = cross_validation(model, X, y, _cv = fold_number)
print(cross_val_results)

plot_result(model_name,
            "Accuracy",
            f"Accuracy scores in {fold_number} Folds",
            cross_val_results["Training Accuracy scores"],
            cross_val_results["Validation Accuracy scores"],
            fold_number)





"""
from sklearn.tree import DecisionTreeClassifier
decision_tree_model = DecisionTreeClassifier(criterion="entropy",
                                     random_state=0)
decision_tree_result = cross_validation(decision_tree_model, X, encoded_y, 5)
print(decision_tree_result)






precision, recall, thresholds = precision_recall_curve(labels, predictions)
auc = metrics.auc(recall, precision)


with open('scores.json', 'w') as fd:
    json.dump({'auc': auc}, fd)


with open('prc.json', 'w') as fd:
    json.dump({'prc': [{
            'precision': p,
            'recall': r,
            'threshold': t
        } for p, r, t in zip(precision, recall, thresholds)
    ]}, fd)



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



"""
