import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import preprocessing

# First stab at a model for the user data and session logs

# Set the current working directory
os.chdir('/Users/ianmacomber/Kaggle Airbnb')

# Bring in the test users data, train users data, session aggregates data.
test_users = pd.read_table('Data/clean_test_users.csv', sep=',')
train_users = pd.read_table('Data/clean_train_users.csv', sep=',')
session_aggregates = pd.read_table('Data/tbltestsessionstmp4.csv', header=-1)

print(test_users.shape)         # (62096, 17)
print(train_users.shape)        # (213451, 18)
print(session_aggregates.shape) # (135484, 47)

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
train = pd.merge(train_users, session_aggregates, how="left", left_on=["id"], right_on=["user_id"])
test = pd.merge(test_users, session_aggregates, how="left", left_on=["id"], right_on=["user_id"])

# We would like our validation set to be only items that have session data.

print(test.shape) # (61668, 64)
print(test_users.shape) # (62096, 17)
print(train.shape) # (213451, 65)
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

X_test.fillna(-1) # Definitely take more looks at this later
X.fillna(-1)      # Take a look at this later

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

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer

def dcg_score(y_true, y_score, k=5):

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    gain = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)

def ndcg_score(ground_truth, predictions, k=5):

    lb = LabelBinarizer()
    T = lb.fit_transform(ground_truth)

    scores = []

    # Iterate over each y_true and compute the DCG score
    for y_true, y_score in zip(T, predictions):
        actual = dcg_score(y_true, y_score, k)
        best = dcg_score(y_true, y_true, k)
        score = float(actual) / float(best)
        scores.append(score)

    return np.mean(scores)

# NDCG Scorer function
ndcg_scorer = make_scorer(ndcg_score, needs_proba=True)

'''
GridSearchCV(cv=5, error_score='raise',
       estimator=XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3,
       min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
       objective='binary:logistic', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=0, silent=True, subsample=1),
       fit_params={}, iid=True, n_jobs=-1,
       param_grid={'n_estimators': [25, 50], 'learning_rate': [0.1, 0.2], 'max_depth': [6, 20]},
       pre_dispatch='2*n_jobs', refit=True,
       scoring=make_scorer(ndcg_score, needs_proba=True), verbose=3)
'''

xgb = GridSearchCV(
    XGBClassifier(seed=0, 
    objective='multi:softprob'
    ),
    param_grid={
        'max_depth': [5, 6],
        'n_estimators': [26,27],
        'learning_rate': [0.1],
        'subsample': [0.45],
        'colsample_bytree': [0.5],
        'gamma': [0.1, 0.15],
        'max_delta_step': [0]
    },
    cv=5,
    verbose=4,
    n_jobs=-1,
    #scoring=make_scorer(ndcg_scorer)
    scoring=ndcg_scorer
    )

# currently - just added gamma to this thing.
# should test it here, should test it on the non-full user model
# then eventually merge these things in 

xgb.best_params_ 
'''
{'colsample_bytree': 0.5,
 'gamma': 0.15,
 'learning_rate': 0.1,
 'max_delta_step': 0,
 'max_depth': 6,
 'n_estimators': 27,
 'subsample': 0.45}
'''

xgb.best_score_ # 0.83585339132974634

xgb = XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=27,
                    objective='multi:softprob', subsample=0.4, colsample_bytree=0.5, seed=0)  

xgb.fit(X, y)
xgb_predictions = xgb.predict_proba(X_test)

# Put these in a good form to spit out
xgb_predictions = xgb_predictions.ravel()

# Have to ensure these are in the same order, yep, looks good
classes = np.tile(xgb.classes_, X_test.shape[0])
ids = np.repeat(test["id"].values, 12)

print(xgb_predictions.shape)
print(classes.shape)
print(ids.shape)
print(test_users['id'].shape)
print(test['id'].shape)

# We want to make this a list of most likely occurances

submission = pd.DataFrame()

submission['id'] = ids
submission['country'] = classes
submission['xgb_prob'] = xgb_predictions
submission['prob'] = submission['xgb_prob']

submission = submission.sort_values(['id', 'prob'], ascending=[True, False])
submission[['id', 'country']].to_csv('Data/airbnb_sessions_xgboost.csv', index=False)

submission.shape