import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# load dataset
dta = sm.datasets.fair.load_pandas().data
# add "affair" column: 1 represents having affairs, 0 represents not. The original column is the measure of time spent in extramarital affairs
dta['affair'] = (dta.affairs > 0).astype(int)


dataforregression=pd.get_dummies(dta, columns =['occupation'])


#specify columns for model FOR NOW JUST SPECIFY AGE
X = dataforregression.iloc[:,1:3]
print(X.head())
#specify column with labels in
y = dataforregression.iloc[:,8]
#drop response column from predictors
#X=X.drop(['affair'],axis=1)
#function form sklearn to split data into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

with open('affairslogistic.pkl', 'wb') as fid:
    pickle.dump(classifier, fid,2) 