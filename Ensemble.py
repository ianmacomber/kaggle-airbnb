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

# First stab at an ensemble model

test_users = pd.read_table('/Users/ianmacomber/Python Work/Kaggle/Airbnb/Data/clean_test_users.csv', sep=',')
train_users = pd.read_table('/Users/ianmacomber/Python Work/Kaggle/Airbnb/Data/clean_train_users.csv', sep=',')
session_aggregates = pd.read_table('/Users/ianmacomber/Python Work/Kaggle/Airbnb/Data/tbltestsessionstmp4.csv', header=-1)

print(test_users.shape)
print(train_users.shape)

session_aggregates.columns = [
'user_id',
'total_clicks',
'total_secs_elapsed',
'total_site_visits_hour',
'total_site_visits_day',
'action_requested',
'action_create',
'action_authenticate',
'action_pending',
'action_travel_plans_current',
'action_personalize',
'action_confirm_email',
'action_payment_instruments',
'action_custom_recommended_destinations',
'action_type_booking_request',
'action_type_submit',
'action_detail_post_checkout_action',
'action_detail_change_trip_characteristics',
'action_detail_your_trips',
'action_detail_wishlist_content_update',
'action_detail_user_profile',
'action_detail_contact_host',
'action_detail_similar_listings',
'action_detail_create_user',
'action_detail_payment_instruments',
'action_detail_login',
'action_detail_book_it',
'action_detail_view_reservations',
'action_detail_trip_availability',
'total_device_types',
'distinct_action',
'distinct_action_type',
'distinct_action_detail'
]

session_aggregates.dtypes

session_aggregates.describe()

# There are some things in the test group that are not in the session aggregates
train = pd.merge(train_users, session_aggregates, how="inner", left_on=["id"], right_on=["user_id"])
test = pd.merge(test_users, session_aggregates, how="inner", left_on=["id"], right_on=["user_id"])

print(test.shape) # (61668, 50)
print(test_users.shape) # (62096, 17)
print(train.shape) # (0, 51)
print(train_users.shape) # (171239, 18)

collist = list(train.columns.values)
collist.remove('id')
collist.remove('country_destination')

X = train[collist]
y = train['country_destination']
X_test = test[collist]

# X_test.fillna(-1) # Definitely take more looks at this later

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
'''
rfc = GridSearchCV(RandomForestClassifier(random_state=79), cv=4, verbose=2, n_jobs=-1,
                  param_grid={"n_estimators": [450, 500],
                              "min_samples_split": [19, 21],
                              "max_features": [19, 21] # This can't go above the total features.pr
                              })

rfc.best_params_ # {'max_features': 19, 'min_samples_split': 19, 'n_estimators': 450}
rfc.best_score_ # 0.69733793944320255
'''

# Generally 10 min a fit
abc = GridSearchCV(AdaBoostClassifier(), cv=4, verbose=2, n_jobs=-1, 
                  param_grid={"n_estimators": [350],
                              "learning_rate": [.1],
                              "base_estimator": [DecisionTreeClassifier(max_depth=23),
                                                 DecisionTreeClassifier(max_depth=25)
                                                 ]
                              })
abc.best_params_
''' This beat things bigger than it
{'base_estimator': DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=23,
             max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             presort=False, random_state=None, splitter='best'),
 'learning_rate': 0.1,
 'n_estimators': 350}
 '''
abc.best_score_ # 0.68198875567296624

# This is pretty well dialed in
rfc = RandomForestClassifier(n_estimators=450, max_features=19, min_samples_split=19, random_state=79)
abc = AdaBoostClassifier(DecisionTreeClassifier(max_depth=23), learning_rate=0.1, n_estimators=400)
etc = ExtraTreesClassifier(min_samples_split=20, max_features=25, n_estimators=500, random_state=79)   

rfc.fit(X, y)
abc.fit(X, y)
etc.fit(X, y)

# Consider using log proba here
rfc_predictions = rfc.predict_proba(X_test)
abc_predictions = abc.predict_proba(X_test)
etc_predictions = etc.predict_proba(X_test)

abc_

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

# We want to make this a list of most likely occurances
missingusers = test_users['id'][~test_users['id'].isin(test['id'])]
missingclasses = np.array(['NDF', 'US', 'other', 'FR', 'IT'])
missingclassvalues = np.array([5, 4, 3, 2, 1])

missinguser_ids = np.repeat(missingusers.values, 5)
missingclasses = np.tile(missingclasses, missingusers.shape[0])
missingclassvalues = np.tile(missingclassvalues, missingusers.shape[0])

print(missinguser_ids.shape)
print(missingclasses.shape)
print(missingclassvalues.shape)

submission = pd.DataFrame()

submission['id'] = np.concatenate((ids, missinguser_ids))
submission['country'] = np.concatenate((classes, missingclasses))
submission['rfc_prob'] = np.concatenate((rfc_predictions, missingclassvalues))
submission['abc_prob'] = np.concatenate((abc_predictions, missingclassvalues))
submission['etc_prob'] = np.concatenate((etc_predictions, missingclassvalues))
submission['prob'] = submission['rfc_prob']*3 + submission['abc_prob'] + submission['etc_prob'] # Something very weird with abc

submission = submission.sort_values(['id', 'prob'], ascending=[True, False])
submission[['id', 'country']].to_csv('airbnb_sessions_ensemble.csv', index=False)

submission.shape

# Next steps
# Look through sessions data, see if I'm missing anything
# Confusion matrix?
# See where one model prefers something over another model
# Test different techniques for the missing sessions data
# Clean up some of the age stuff and other variables?


