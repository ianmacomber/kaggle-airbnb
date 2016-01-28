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
* `user_model.py` - This is a model containing ALL of the train user data, but none of the session user data.  The training set has 213,451 rows and 18 columns, spanning a longer time period.
* `session_user_model.py` - This is a model containing ALL of the session level data, but only a subset of the user data.  The training set has 73,815 rows and 65 columns.  This is my current best implimentation.
* `session_user_model_xgboost.py` - This is a model using the xgboost package.

## Current State

Score of ~~0.87758~~ 0.87782 using variations of random and extra tree classifiers.
Only using the subset of training data that has corresponding session level data.

Score of 0.85620 using random tree classifiers for the entire data set (no session level)

Score of ~~0.87660~~ 0.87631 on xgboost with training data that has corresponding session level data.

## To-do

* ~~Incorporate a model that doesn't use session level data, so I can use the whole training set~~
* ~~Try to cross validate using NDCG as a score.  Currently using default score.~~
* ~~Check out xgboost~~
* Look at other classification methods (SVC, KNN)
* Improve feature engineering, especially of session-level data
* Investigate seasonal trends.  Test set is only July-Sept.
* Confusion matrix
* When do user models make different predictions?
* Ensemble of models
* Best test set of train data?  Maybe instead of CV, use a smart choice of specific split to incorporate session and non-session level training data?  
We don't really care if a model does a good job of predicting non-session data, we are only prediction on session data.
* Combine old data with new data using left outer join instead of inner, then setting to min?
* ~~Investigate log-loss as a score?~~
* Tune XGBoost with standard methods, deal with NA's, restrict features
* Test a model that uses all available data and treats missing sessions as -1
