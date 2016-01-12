# Airbnb New User Bookings

This is a code repo for the [Airbnb Kaggle Competition](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings)

In this recruiting competition, Airbnb challenges you to predict in which country a new user will make his or her first booking.

## Airbnb File Descriptions

These files can be found through the website and are stored in a `/Data` folder that is not committed to git.

* `train_users.csv` - the training set of users
* `test_users.csv` - the test set of users
* `sessions.csv` - web sessions log for users
* `countries.csv` - summary statistics of destination countries in this dataset and their locations
* `age_gender_bkts.csv` - summary statistics of users' age group, gender, country of destination

## My File Descriptions

* `Kaggle Overview.py` - contains a general overview of the user data
* `Sessions Overview.py` - contains a general overview of just the sessions data
* `users_data_clean.py` - cleans the user data and turns it into `clean_test_users.csv` and `clean_train_users.csv`
* `sessions_feature_extraction.hql` - file TBD, cleans the sessions table to turn it into user level variables that can be used.  Generates `tbltestsessionstmp4.csv` which may certainly have a new name in the future.
* `Ensemble2.py` - current best implimentation

## Current State

Score of 0.87758 using variations of random and extra tree classifiers.
Only using the subset of training data that has corresponding session level data.

## To-do

* Incorporate a model that doesn't use session level data, so I can use the whole training set.
* Try to cross validate using NDCG as a score.  Currently using default score.
* Check out xgboost.
* Look at other classification methods (SVM, KNN)
* Improve feature engineering, especially of session-level data
* Investigate seasonal trends.  Test set is only July-Sept
