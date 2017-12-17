from flask import Flask, request, render_template
from sklearn.externals import joblib
import pandas as pd
import numpy
import logging
from logging.handlers import RotatingFileHandler
import json

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')


@app.route('/getresponse',methods=['POST','GET'])
def getresponse():
	if request.method=='POST':
		

		form_values=request.form.to_dict()
		app.logger.info(form_values)
		classifier = joblib.load('models/model.pkl')
		model_columns = joblib.load('models/model_columns.pkl')
		app.logger.info("Taking post request form details and cleaning to match model input"),
		#take the json and turn into a pandas data frame
		d=[{'age':form_values['age'],'yrs_married':form_values['yrs_married']}]
		query_df = pd.DataFrame(data=d)
		#get dummies for any categorical variables
		#query = pd.get_dummies(query_df)

		print(query_df)
		#print(model_columns.columns)




		for col in model_columns:
			if col not in query_df.columns:
				query_df[col] = 0

		app.logger.info("Predicting outcome response given inputs")

		prediction = classifier.predict(query_df)
		app.logger.info('RESPONSE PREDICTION  = %s',str(prediction))
		return render_template('result.html',prediction=prediction)



if __name__ == '__main__':

	app.run(debug=True,host='0.0.0.0')
	LOGGING_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
	LOGGING_LEVEL  = logging.DEBUG
	formatter = logging.Formatter(app.config['LOGGING_FORMAT'])
	handler.setFormatter(formatter)
	app.logger.addHandler(handler)