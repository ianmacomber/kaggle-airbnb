# Kaggle Users Data Clean
# I'm going to break out my python files based on what I use them for
# This is where I clean all my user data

'''
In this challenge, you are given a list of users along with their demographics, web session records, 
and some summary statistics. You are asked to predict which country a new user's first booking destination 
will be. All the users in this dataset are from the USA.

There are 12 possible outcomes of the destination country: 
'US', 'FR', 'CA', 'GB', 'ES', 'IT', 'PT', 'NL','DE', 'AU', 'NDF' (no destination found), and 'other'. 
Please note that 'NDF' is different from 'other' because 'other' means there was a booking, 
but is to a country not included in the list, while 'NDF' means there wasn't a booking.

The training and test sets are split by dates. 
In the test set, you will predict all the new users with first activities after 4/1/2014. 
In the sessions dataset, the data only dates back to 1/1/2014, while the users dataset dates back to 2010. 
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

test_users = pd.read_table('/Users/ianmacomber/Python Work/Kaggle/Airbnb/Data/test_users.csv', sep=',')
train_users = pd.read_table('/Users/ianmacomber/Python Work/Kaggle/Airbnb/Data/train_users_2.csv', sep=',')

# Gender -- make things only Male or Female randomly -- actually, let's make this 
'''
i = 0
def get_gender(gender):
    global i
    if gender != 'FEMALE' and gender != 'MALE':
        return 'FEMALE' if(i % 2) else 'MALE'
    i = i + 1
    return gender
'''

def get_gender(gender):
    if gender == 'OTHER':
        return '-unknown-'
    else:
        return gender

# replace all values other than 'FEMALE' and 'MALE'
train_users['gender'] = train_users['gender'].apply(get_gender)
test_users['gender'] = test_users['gender'].apply(get_gender)

# Age is fucked up
def age_cleaner(x):
    if x <= 150:
        return x
    elif x > 150 and x < 1995:
        return 2014-x
    else:
        return np.nan

train_users['age'] = train_users['age'].apply(lambda x: age_cleaner(x))
test_users['age'] = test_users['age'].apply(lambda x: age_cleaner(x))
train_users['age'] = train_users['age'].fillna(train_users['age'].median())
test_users['age'] = test_users['age'].fillna(train_users['age'].median())

# Only wrinkle for signup_method is that test_users have a google one with very few.  Gonna make these basic.
def signup_method_cleaner(x):
    if x in ['google', 'weibo']:
        return 'basic'
    else:
        return x

test_users['signup_method'] = test_users['signup_method'].apply(signup_method_cleaner)

# Setting the ones not shared by test and train to 0
def signup_flow_cleaner(x):
    if x in [25, 24, 23, 12, 21]:
        return x
    else:
        return 0

train_users['signup_flow'] = train_users['signup_flow'].apply(signup_flow_cleaner).astype('str')
test_users['signup_flow'] = test_users['signup_flow'].apply(signup_flow_cleaner).astype('str')

# Getting rid of languages not seen in train
def language_cleaner(x):
    if x in ['-unknown-', 'id']:
        return 'en'
    else:
        return x

train_users['language'] = train_users['language'].apply(language_cleaner)
test_users['language'] = test_users['language'].apply(language_cleaner)

# Not touching affiliate_channel
# train_users.groupby(['affiliate_channel'])['affiliate_channel'].count().sort_values(ascending=False).head(25)
# test_users.groupby(['affiliate_channel'])['affiliate_channel'].count().sort_values(ascending=False).head(25) 

def affiliate_provider_cleaner(x):
    if x in ['vast', 'padmapper', 'facebook-open-graph', 'yahoo', 'gsp', 'meetup', 'email-marketing', 'naver', 'baidu', 'wayn', 'yandex', 'craigslist', 'daum', 'meetup']:
        return 'direct'
    else:
        return x

train_users['affiliate_provider'] = train_users['affiliate_provider'].apply(affiliate_provider_cleaner)
test_users['affiliate_provider'] = test_users['affiliate_provider'].apply(affiliate_provider_cleaner)

def first_affiliate_tracked_cleaner(x):
    if x in ['product', 'tracked-other', 'marketing', 'local ops']:
        return 'untracked'
    else:
        return x 

train_users['first_affiliate_tracked'] = train_users['first_affiliate_tracked'].fillna('product').apply(first_affiliate_tracked_cleaner)
test_users['first_affiliate_tracked'] = test_users['first_affiliate_tracked'].fillna('product').apply(first_affiliate_tracked_cleaner)    

# Not touching signup_app
# train_users.groupby(['signup_app'])['signup_app'].count().sort_values(ascending=False).head(25)
# test_users.groupby(['signup_app'])['signup_app'].count().sort_values(ascending=False).head(25) 

# Not touching first_device_type
# train_users.groupby(['first_device_type'])['first_device_type'].count().sort_values(ascending=False).head(25)
# test_users.groupby(['first_device_type'])['first_device_type'].count().sort_values(ascending=False).head(25) 

def first_browser_cleaner(x):
    if x in ['Chrome', '-unknown-', 'Safari', 'Mobile Safari', 'Firefox', 'IE']:
        return x
    else:
        return '-unknown-'

train_users['first_browser'] = train_users['first_browser'].apply(first_browser_cleaner)
test_users['first_browser'] = test_users['first_browser'].apply(first_browser_cleaner)

# train_users.groupby(['first_browser'])['first_browser'].count().sort_values(ascending=False).head(25)
# test_users.groupby(['first_browser'])['first_browser'].count().sort_values(ascending=False).head(25) 

# Let's investigate some pandas dateparsing and put these into a format that's useable
# Turn these into timestamps

train_users['date_account_created'] = pd.to_datetime(train_users['date_account_created'])
test_users['date_account_created'] = pd.to_datetime(test_users['date_account_created'])

# The timestamp_first_active is in a weird int format
# 20100104032827
# a = 2010-01-04 03:28:27
# I'd like to turn this into a string, then turn it back into a timestamp.

train_users['timestamp_first_active'] = train_users['timestamp_first_active'].apply(str)
test_users['timestamp_first_active'] = test_users['timestamp_first_active'].apply(str)
train_users['timestamp_first_active'] = pd.to_datetime(train_users['timestamp_first_active'])
test_users['timestamp_first_active'] = pd.to_datetime(test_users['timestamp_first_active'])

# I bet that hour of day and day of week is an important part of this

# Using the timestamp functions here to break a date into its important parts, then dropping the column.
train_users['month_account_created'] = train_users['date_account_created'].apply(lambda x: x.month)
train_users['weekday_account_created'] = train_users['date_account_created'].apply(lambda x: x.weekday())
test_users['month_account_created'] = test_users['date_account_created'].apply(lambda x: x.month)
test_users['weekday_account_created'] = test_users['date_account_created'].apply(lambda x: x.weekday())
train_users.drop(['date_account_created'], axis=1,inplace=True)
test_users.drop(['date_account_created'], axis=1,inplace=True)

# Date first booking is no longer important
train_users.drop(['date_first_booking'], axis=1,inplace=True)
test_users.drop(['date_first_booking'], axis=1,inplace=True)

# First active still is
train_users['month_first_active'] = train_users['timestamp_first_active'].apply(lambda x: x.month)
train_users['hour_first_active'] = train_users['timestamp_first_active'].apply(lambda x: x.hour)
train_users['weekday_first_active'] = train_users['timestamp_first_active'].apply(lambda x: x.weekday())
test_users['month_first_active'] = test_users['timestamp_first_active'].apply(lambda x: x.month)
test_users['hour_first_active'] = test_users['timestamp_first_active'].apply(lambda x: x.hour)
test_users['weekday_first_active'] = test_users['timestamp_first_active'].apply(lambda x: x.weekday())
train_users.drop(['timestamp_first_active'], axis=1,inplace=True)
test_users.drop(['timestamp_first_active'], axis=1,inplace=True)

print(train_users.shape)
print(test_users.shape)
print(train_users.dtypes)
print(test_users.dtypes)

collist = list(train_users.columns.values)
collist.remove('id')
collist.remove('country_destination')

'''
# This gives me all of the top values in a group
for i in collist:
    print(train_users.groupby([i])[i].count().sort_values(ascending=False).head(30))

# This gives me all of the items in my training set not in the test set
for i in collist:
    for n in train_users[i].unique():
        if not np.in1d(n, test_users[i].unique()):
           print(i, n, np.in1d(n, test_users[i].unique()))

# This gives me all of the items in my test set not in the training set
for i in collist:
    for n in test_users[i].unique():
        if not np.in1d(n, train_users[i].unique()):
           print(i, n, np.in1d(n, train_users[i].unique()))
'''

X = train_users[collist]
y = train_users['country_destination']
X_test = test_users[collist]

# Write these to .csv
train_users.to_csv('/Users/ianmacomber/Python Work/Kaggle/Airbnb/Data/clean_train_users.csv', index=False)
test_users.to_csv('/Users/ianmacomber/Python Work/Kaggle/Airbnb/Data/clean_test_users.csv', index=False)

# This is all secondary, just testing out models.

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

# Running a random forest classifier
rfc = RandomForestClassifier(n_estimators=200, max_features=6, min_samples_split=9)

rfc = GridSearchCV(RandomForestClassifier(), cv=4, verbose=2, n_jobs=-1,
                  param_grid={"n_estimators": [200],
                              "min_samples_split": [8,9],
                              "max_features": [5, 6] # This can't go above the total features.pr
                              })

rfc.fit(X, y)

rfc.best_params_ # {'max_features': 6, 'min_samples_split': 9, 'n_estimators': 200}

zip(X.columns, rfc.feature_importances_)

predictions = rfc.predict_proba(X_test)

# Put these in a good form to spit out
predictions = predictions.ravel()
classes = np.tile(rfc.classes_, X_test.shape[0])
ids = np.repeat(test_users["id"].values, 12)

print(predictions.shape)
print(classes.shape)
print(ids.shape)

submission = pd.DataFrame()

submission['id'] = ids
submission['country'] = classes
submission['prob'] = predictions

submission = submission.sort_values(['id', 'prob'], ascending=[True, False])
submission[['id', 'country']].to_csv('airbnb.csv', index=False)

# This gets me to .929!  Leader is .933.
# Next ideas -- insure that NDF booking has the most likely things behind it and doublecheck.
# Use ravel or something like that for better query submission etc.
# Start generating query level session metrics
# Look at other classfiers, ensemble methods
# Keep tuning random forest classifier? -- Probably not this, it's close to good.

