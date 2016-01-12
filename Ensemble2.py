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
print(session_aggregates.shape)

session_aggregates.columns = [
'user_id',
'totalclicks',
'total_secs_elapsed',
'total_site_visits_sep_min',
'total_site_visits_sep_hour',
'total_site_visits_sep_day',
'action_requested',
'action_create',
'action_authenticate',
'action_pending',
'action_travel_plans_current',
'action_personalize',
'action_confirm_email',
'action_payment_instruments',
'action_custom_recommended_destinations',
'action_languages_multiselect',
'action_type_booking_request',
'min_visit_seq_action_type_booking_request',
'min_total_secs_elapsed_action_type_booking_request',
'clicks_after_action_type_booking_request',
'seconds_after_action_type_booking_request',
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
'min_visit_seq_action_detail_book_it',
'min_total_secs_elapsed_action_detail_book_it',
'action_detail_view_reservations',
'action_detail_trip_availability',
'action_detail_instant_book',
'action_detail_user_languages',
'action_detail_p5',
'action_detail_p4',
'action_detail_p3',
'action_detail_message_to_host',
'totaldevicetypes',
'distinct_action',
'distinct_action_type',
'distinct_action_detail'
]

print(session_aggregates.dtypes)

session_aggregates['min_visit_seq_action_type_booking_request'] = session_aggregates['min_visit_seq_action_type_booking_request'].fillna(-1).astype(int)
session_aggregates['min_total_secs_elapsed_action_type_booking_request'] = session_aggregates['min_total_secs_elapsed_action_type_booking_request'].fillna(-1).astype(int)
session_aggregates['clicks_after_action_type_booking_request'] = session_aggregates['clicks_after_action_type_booking_request'].fillna(-1).astype(int)
session_aggregates['seconds_after_action_type_booking_request'] = session_aggregates['seconds_after_action_type_booking_request'].fillna(-1).astype(int)
session_aggregates['min_visit_seq_action_detail_book_it'] = session_aggregates['min_visit_seq_action_detail_book_it'].fillna(-1).astype(int)
session_aggregates['min_total_secs_elapsed_action_detail_book_it'] = session_aggregates['min_total_secs_elapsed_action_detail_book_it'].fillna(-1).astype(int)

# There are some things in the test group that are not in the session aggregates
train = pd.merge(train_users, session_aggregates, how="inner", left_on=["id"], right_on=["user_id"])
test = pd.merge(test_users, session_aggregates, how="inner", left_on=["id"], right_on=["user_id"])

print(test.shape) # (61668, 64)
print(test_users.shape) # (62096, 17)
print(train.shape) # (73815, 65)
print(train_users.shape) # (213451, 18)

collist = list(train.columns.values)
collist.remove('id')
collist.remove('user_id')
collist.remove('country_destination')

''' Gonna take these out now for speed
collist.remove('min_total_secs_elapsed_action_detail_book_it')
collist.remove('action_detail_post_checkout_action')
collist.remove('min_visit_seq_action_detail_book_it')
collist.remove('action_payment_instruments')
collist.remove('action_detail_payment_instruments')
collist.remove('action_detail_trip_availability')
collist.remove('action_detail_book_it')
collist.remove('action_detail_instant_book')
collist.remove('action_detail_user_languages')
collist.remove('action_detail_view_reservations')
collist.remove('action_custom_recommended_destinations')
'''

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
                              "min_samples_split": [26, 30, 34],
                              "max_features": [16, 20, 24] # This can't go above the total features.pr
                              })

rfc.best_params_ # {'max_features': 20, 'min_samples_split': 30, 'n_estimators': 450}
rfc.best_score_ # 0.69815078236130867
'''

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

abc.best_score_ # 0.68198875567296624
'''


etc = GridSearchCV(ExtraTreesClassifier(random_state=79), cv=4, verbose=2, n_jobs=-1,
                  param_grid={"n_estimators": [600, 700],
                              "max_features": [33], # This can't go above the total features
                              "min_samples_split": [28, 30, 32],
                              "bootstrap": [True]
                              })
                              
etc.best_params_ # {'bootstrap': True, 'max_features': 33, 'min_samples_split': 30, 'n_estimators': 600}
etc.best_score_ # 69279956648377705




# This is pretty well dialed in
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
submission['prob'] = submission['rfc_prob']*2 + submission['etc_prob'] # Something very weird with abc

submission['prob'] = submission['etc_prob']

submission = submission.sort_values(['id', 'prob'], ascending=[True, False])
submission[['id', 'country']].to_csv('airbnb_sessions_ensemble.csv', index=False)

submission.shape

# Next steps
# Look through sessions data, see if I'm missing anything
# Confusion matrix?
# See where one model prefers something over another model
# Test different techniques for the missing sessions data
# Clean up some of the age stuff and other variables?


