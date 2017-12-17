from flask import Flask, request, render_template
from sklearn.externals import joblib
import pandas as pd
import numpy


app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')


@app.route('/getresponse',methods=['POST','GET'])
def getresponse():
	if request.method=='POST':
		form_values=request.json

		classifier = joblib.load('model.pkl')
		model_columns = joblib.load('model_columns.pkl')
		#hardcode values for now
		form_values=[{'age':45,'yrs_married':10}]
		#take the json and turn into a pandas data frame
		query_df = pd.DataFrame(form_values)
		#get dummies for any categorical variables
		query = pd.get_dummies(query_df)
		print('VECTOR FOR PREDICTION')
		print(query)


		for col in model_columns:
			if col not in query.columns:
				query[col] = 0

		print(query)

		prediction = classifier.predict(query)
		return render_template('result.html',prediction=prediction)



if __name__ == '__main__':

	app.run(debug=True)