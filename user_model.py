import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

# First stab at a model for the user data only
# Purpose of this is more user data, because we aren't throwing out
# user data that doesn't have any session level detail

# Set the current working directory
os.chdir('/Users/ianmacomber/Kaggle Airbnb')

test = pd.read_table('Data/clean_test_users.csv', sep=',')
train = pd.read_table('Data/clean_train_users.csv', sep=',')

print(test.shape) # (62096, 17)
print(train.shape) # (213451, 18)

collist = list(train.columns.values)
collist.remove('id')
collist.remove('country_destination')

X = train[collist]
y = train['country_destination']
X_test = test[collist]

for f in X.columns:
    if X[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        if f not in X_test.columns:
            lbl.fit(np.unique(list(X[f].values)))
            X[f] = lbl.transform(list(X[f].values))
        else:
            lbl.fit(np.unique(list(X[f].values) + list(X_test[f].values)))
            X[f] = lbl.transform(list(X[f].values))
            X_test[f] = lbl.transform(list(X_test[f].values))
            

rfc = GridSearchCV(RandomForestClassifier(random_state=79), cv=4, verbose=2, n_jobs=-1,
                  param_grid={"n_estimators": [400, 500],
                              "min_samples_split": [26, 34],
                              "max_features": [16, 24] # This can't go above the total features.pr
                              })

rfc.best_params_ # 
rfc.best_score_ #


# Generally 10 min a fit
'''
abc = GridSearchCV(AdaBoostClassifier(), cv=4, verbose=2, n_jobs=-1, 
                  param_grid={"n_estimators": [350],
                              "learning_rate": [.1],
                              "base_estimator": [DecisionTreeClassifier(max_depth=23),
                                                 DecisionTreeClassifier(max_depth=25)
                                                 ]
                              })
abc.best_params_
'''

''' This beat things bigger than it
{'base_estimator': DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=23,
             max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             presort=False, random_state=None, splitter='best'),
 'learning_rate': 0.1,
 'n_estimators': 350}

abc.best_score_ #
'''


etc = GridSearchCV(ExtraTreesClassifier(random_state=79), cv=4, verbose=2, n_jobs=-1,
                  param_grid={"n_estimators": [600, 700],
                              "max_features": [33], # This can't go above the total features
                              "min_samples_split": [28, 30, 32],
                              "bootstrap": [True]
                              })
                              
etc.best_params_ # {'bootstrap': True, 'max_features': 33, 'min_samples_split': 30, 'n_estimators': 600}
etc.best_score_ # 

# This is not dialed in at all
rfc = RandomForestClassifier(n_estimators=450, max_features=20, min_samples_split=30, random_state=79)
abc = AdaBoostClassifier(DecisionTreeClassifier(max_depth=23), learning_rate=0.1, n_estimators=400)
etc = ExtraTreesClassifier(min_samples_split=35, max_features=30, n_estimators=500, random_state=79)   

'''
rfc.fit(X, y)
abc.fit(X, y)
etc.fit(X, y)
'''

sorted(dict(zip(X.columns, rfc.feature_importances_)).items(), key=lambda x: x[1], reverse=True)
sorted(dict(zip(X.columns, abc.feature_importances_)).items(), key=lambda x: x[1], reverse=True)
sorted(dict(zip(X.columns, etc.feature_importances_)).items(), key=lambda x: x[1], reverse=True)

# Consider using log proba here
rfc_predictions = rfc.predict_proba(X_test)
abc_predictions = abc.predict_proba(X_test)
etc_predictions = etc.predict_proba(X_test)

# Put these in a good form to spit out
rfc_predictions = rfc_predictions.ravel()
abc_predictions = abc_predictions.ravel()
etc_predictions = etc_predictions.ravel()
# Have to ensure these are in the same order, yep, looks good
classes = np.tile(rfc.classes_, X_test.shape[0])
ids = np.repeat(test["id"].values, 12)

print(rfc_predictions.shape)
print(abc_predictions.shape)
print(etc_predictions.shape)
print(classes.shape)
print(ids.shape)
print(test_users['id'].shape)
print(test['id'].shape)

submission = pd.DataFrame()

submission['id'] = np.concatenate((ids, missinguser_ids))
submission['country'] = np.concatenate((classes, missingclasses))
submission['rfc_prob'] = np.concatenate((rfc_predictions, missingclassvalues))
submission['abc_prob'] = np.concatenate((abc_predictions, missingclassvalues))
submission['etc_prob'] = np.concatenate((etc_predictions, missingclassvalues))
submission['prob'] = submission['rfc_prob'] # + submission['etc_prob'] # Something very weird with abc

submission['prob'] = submission['etc_prob']

submission = submission.sort_values(['id', 'prob'], ascending=[True, False])
submission[['id', 'country']].to_csv('Data/airbnb_sessions_ensemble.csv', index=False)

submission.shape

# Next steps
# Look through sessions data, see if I'm missing anything
# Confusion matrix?
# See where one model prefers something over another model
# Test different techniques for the missing sessions data
# Clean up some of the age stuff and other variables?


