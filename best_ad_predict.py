import os
import pandas as pd
from sklearn.cross_validation import cross_val_score
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from dateutil import parser
import numpy as np

def process_csv(csv_path, test=False):
	if not test:
		columns = ['Timestamp', 'Hour', 'Ad', 'Browser', 'Platform', 'Region', 'Clicked']
	else:
		columns = ['Timestamp', 'Hour', 'Browser', 'Platform', 'Region']

	df = pd.read_csv(csv_path, names=columns)


	## -- Feature Engineering from Timestamp -- ## 
	#df['Hour'] = df['Timestamp'].apply(lambda x: round((float(x.split(':')[0][-2:]) + float(x.split(':')[1])/60), 1))
	df['Day_of_week'] = df['Timestamp'].apply(lambda x: parser.parse(x).strftime("%A"))
	df['Day'] = df['Timestamp'].apply(lambda x: int(x.split('-')[2][:2]))
	df['Year'] = df['Timestamp'].apply(lambda x: int(x.split('-')[0]))
	df['Month'] = df['Timestamp'].apply(lambda x: int(x.split('-')[1]))

	#print df.head(2)
	return df


def train_xgboost(df_train, features, target, save_model=False, cv=False):

	# Numerate feature strings for modeling and save feature_ids.
	feature_ids = {}
	for col in  features + ['Ad']:
		if df_train[col].dtype == "object":
			catigories = list(df_train[col].unique())
			df_train[col] = df_train[col].apply(lambda cat: catigories.index(cat))
			feature_ids[col] = {cat: catigories.index(cat) for cat in catigories}


	# Fit Gradient Boosted decision model.
	X = df_train[features + ['Ad']].as_matrix()
	y = df_train[target].as_matrix()

	
        
    # Declare XGboost model.
	xgb = XGBClassifier(
		learning_rate =0.8,
		n_estimators=54,
		max_depth=5,
		min_child_weight=1,
		gamma=0.2,
		subsample=0.8,
		colsample_bytree=0.75,
		reg_alpha = 15.25,
		objective = 'binary:logistic')
	
	# Fit the model on the data
	xgb.fit(X, y, eval_metric='auc')

	# Print 5-fold cross validation scores if cv=True. 
	if cv: 
		cv_scores = cross_val_score(xgb, X, y, cv=5)
		print 'cross_val_scores:', cv_scores, cv_scores.mean()

	return xgb, feature_ids



def csv_predict(csv_path, model, features, feature_ids):

	# Open CSV into a Pandas dataframe & drop unused features.
	df_test = process_csv(csv_path, test=True)

	# Numerate features from feature_ids dictionary for modeling.
	for feat in features:
		if df_test[feat].dtype == "object":
			df_test[feat] = df_test[feat].map(feature_ids[feat])
	
	# Set X to features matrix, excluding 'Ad'.
	X = df_test[features].as_matrix()
	
	probs_matrix = np.array([]).reshape(X.shape[0], 0)

	# Create matrix of Ad-Click probabilities.
	for ad in range(len(feature_ids['Ad'])):
		pred_proba = model.predict_proba(np.append(X, np.ones((X.shape[0], 1))*ad, axis=1))[:,1]
		probs_matrix  = np.column_stack((probs_matrix, pred_proba))

	# Choose the Best Ad by selecting the ad with the highest click probability.
	df_test['Ad'] = np.argmax(probs_matrix, axis=1)
	
	# Convert categorical features back into strings.
	for feat in feature_ids.keys():
		reversed_dic = {v: k for k, v in feature_ids[feat].iteritems()}
		df_test[feat] = df_test[feat].map(reversed_dic)

	return df_test


def STDIN_predict(raw_in, model, feature_ids):

	# Sets X from raw input.
	X = rawfeatures_2_list(raw_in, features, feature_ids)
	

	# Creates an array of predicted probabilities of click-through for every Ad.
	predicts_array = np.array([]).reshape(-1, 1)
	for ad in range(len(feature_ids['Ad'])):
		 pred = model.predict_proba([X + [ad]])[:,1]
		 predicts_array = np.append(predicts_array, pred)

	# Picks Ad with the best probability of a click through and converts it back to a string.
	ad_ids = {v: k for k, v in feature_ids['Ad'].iteritems()}
	return raw_in + ',' + ad_ids[np.argmax(predicts_array)]


def STDIN_run(output_csv, model, feature_ids):
	print '(Type q to quit).'
	print "Enter the Ad opportunity as below or path to csv:"
	print 'Timestamp,Hour,Browser,Platform>,Region  or  example.csv'
	
	with open(output_csv, "a") as csv_file:
		raw_in = ''
		while (raw_in.lower() != 'q'):
			raw_in = raw_input('->')

			if raw_in.lower() == 'q':
				break
			
			elif raw_in.lower()[-4:] == '.csv':
				csv_predict(raw_in, model, features, feature_ids).to_csv(output_csv, index=False, header=False)
				print 'Ad preditions saved to {}'.format(output_csv)
			
			elif raw_in.count(',') == 4:
				pred_str = STDIN_predict(raw_in, model, feature_ids)
				csv_file.write(pred_str+'\n')
				print "('{}' saved to {})\n".format(pred_str, output_csv)
			
			else:
				print 'Incorrect input.'


def load_model(train_csv, model_pkl, cv=False):
	# Loads pickled model if it exists otherwise trains model from csv.
	if os.path.isfile(model_pkl):	
		print "Loaded {}".format(model_pkl)
		with open('model.pkl', 'rb') as pickle:
			return cPickle.load(pickle)
		
	elif os.path.isfile(train_csv):
		print "Trained model from {} and save as {}".format(train_csv, model_pkl)
		return model_from_csv(train_csv, features, model_pkl, save_model=False, cv=cv)
	
	else:
		print "{} or {} needs to be in the file directory.".format(train_csv, model_pkl)


def rawfeatures_2_list(raw_in, features, feature_ids):
	# Creates and numerate array with features in the correct index positions. 
	
	# Convert raw_in into input_dict and create new features.
	input_dict = {}
	input_dict['Timestamp'] = raw_in.split(':')[0]
	input_dict['Day_of_week'] = parser.parse(input_dict['Timestamp']).strftime("%A")
	input_dict['Day'] = int(input_dict['Timestamp'].split('-')[2][:1])
	input_dict['Year'] = int(input_dict['Timestamp'].split('-')[0])
	input_dict['Month'] = int(input_dict['Timestamp'].split('-')[1])
	input_dict['Hour'] =  int(raw_in.split(':')[0][-2:])
	input_dict['Browser'] =  raw_in.split(',')[2]
	input_dict['Platform'] =  raw_in.split(',')[3]
	input_dict['Region'] =  raw_in.split(',')[4]

	
	# Declare output_list.
	output_list = []

	# Numerize selected features in order using feature_ids and appended to output_array.
	for feat in features:
		if type(input_dict[feat]) == int or type(input_dict[feat]) == float:
			output_list.append(input_dict[feat])
		else:
			output_list.append(feature_ids[feat][input_dict[feat]])

	return output_list

if __name__ == '__main__':
	
	# Parameters
	train_csv = 'training.csv'
	model_pkl = 'model.pkl'
	output_csv = 'output.csv'
	cross_validate = True
	
	# Declare Target
	target = 'Clicked'
	exclude = ['Year', 'Ad']
	
	# Open CSV into a Pandas dataframe & drops unused features.
	df_train = process_csv(train_csv)

	# Choose all predictors/features  except target & exclude.
	# 	Only predictors 'Hour' and 'Region' are the only features that improved performance on cross validation.
	features = [x for x in df_train.columns if x not in [target] + exclude]
	features = ['Hour', 'Region', 'Day', 'Month'] #['Timestamp', 'Hour', 'Browser', 'Platform', 'Region', 'Day_of_week', 'Day', 'Year', 'Month']
	print features

	# Load and/or train model, predict best ad via STDIN. 
	model, feature_ids = train_xgboost(df_train, features, target, save_model=False, cv=cross_validate)
	STDIN_run(output_csv, model, feature_ids)


