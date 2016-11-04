import os
#import sys
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
import cPickle
import numpy as np
from collections import OrderedDict

def process_csv(csv_path, test=False):
	columns = ['Timestamp', 'Hour', 'Ad', 'Browser', 'Platform', 'Region', 'Clicked']
	
	### TEMPORARY ##
	#if test: columns.remove('Ad')
	

	df = pd.read_csv(csv_path, names=columns)

	# Create 'Hour' from 'Timestamp' as a float to improve accuracy.
	df['Hour'] = df['Timestamp'].apply(lambda x: round((float(x.split(':')[0][-2:]) + \
	    float(x.split(':')[1])/60), 2))

	###### TEMPORARY ####
	if test: 
		try: 
			df.drop(['Ad', 'Clicked'], axis=1, inplace=True)
		except:
			pass

	return df


def model_from_csv(csv_path, features, model_pkl, save_model=False, cv=False):
	
	# Open CSV into a Pandas dataframe & drop unused features.
	df_train = process_csv(csv_path)

	# Numerize feature strings for modeling and save feature_ids.
	feature_ids = {}
	for col in  features + ['Ad']:
		catigories = list(df_train[col].unique())
		df_train[col] = df_train[col].apply(lambda cat: catigories.index(cat))
		feature_ids[col] = {cat: catigories.index(cat) for cat in catigories}


	# Fit Gradient Boosted decision model.
	X = df_train[features + ['Ad']]
	y = df_train['Clicked']

	model = GradientBoostingClassifier(n_estimators=37, max_depth=5)
	model.fit(X, y)

	# Print 5-fold cross validation scores if cv=True. 
	if cv: print 'cross_val_scores:', cross_val_score(model, X, y, cv=5)
	
	# Pickel and save model.
	if save_model:
		with open(model_pkl, 'wb') as pickle:
			cPickle.dump((model, feature_ids), pickle)

	return model, feature_ids


def csv_predict(csv_path, model, features, feature_ids):

	# Open CSV into a Pandas dataframe & drop unused features.
	df_test = process_csv(csv_path, test=True)

	# Numerize features from feature_ids dictionary for modeling.
	for feat in features:
		df_test[feat] = df_test[feat].map(feature_ids[feat])
	

	X = df_test[features].as_matrix()
	
	probs_matrix = np.array([]).reshape(X.shape[0], 0)

	# Create maxtix of Ad-Click probabilities.
	for ad in range(len(feature_ids['Ad'])):
		probs_matrix  = np.column_stack((probs_matrix, model.predict_proba(np.append(X, np.ones((X.shape[0], 1))*ad, axis=1))[:,1]))

	# Choose the Best Ad by selecting the ad with the highest click probability.
	ad_ids = {v: k for k, v in feature_ids['Ad'].iteritems()}
	df_test['Best_ad'] = np.argmax(probs_matrix, axis=1)
	df_test['Best_ad'] = df_test['Best_ad'].map(ad_ids)


	return df_test


def STDIN_predict(raw_in, model, feature_ids):

	string_2_array(raw_in, features, feature_ids)

	
	# Set X from raw input.
	X = string_2_array(raw_in, features, feature_ids)
	probs_arry = np.array([])

	# Predict the probabibility of a click for every Ad.
	for ad in range(len(feature_ids['Ad'])):
		 pred = model.predict_proba(np.append(X, np.ones(1)*ad))[:,1]
		 probs_arry = np.append(probs_arry, pred)

	# Pick the best Ad based of probability of click.
	best_ad_str = feature_ids['Ad'][np.argmax(probs_arry)]
	print best_ad_str
	return raw_in + ',' + best_ad_str

def STDIN_run(output_csv, model, feature_ids):
	print '(Type q to quit).'
	print "Enter the Ad opertunity as below or path to csv:"
	print '<Timestamp>, <hour>, <browser>, <platform>, <region>  or  example.csv'
	
	with open(output_csv, "w") as csv_file:
		raw_in = ''
		while (raw_in.lower() != 'q'):
			raw_in = raw_input('->')

			if raw_in.lower() == 'q':
				break
			
			elif raw_in.lower()[-4:] == '.csv':
				#csv_predict(raw_in, model, features, feature_ids)
				print 'Ad preditions saved to {}'.format(raw_in[:-5]+'_pred.csv')
			
			elif raw_in.count(',') == 4:
				pred_str = STDIN_predict(raw_in, model, feature_ids)
				csv_file.write(pred_str+'\n')
				print "'{}' Saved to {}".format(pred_str, output_csv)
			
			else:
				print 'Incorrect input.'

def load_model(train_csv, model_pkl):
	# Loads pickled model if it exsists otherwise trains model from csv.
	if os.path.isfile(model_pkl):	
		print "Loaded {}".format(model_pkl)
		with open('model.pkl', 'rb') as pickle:
			return cPickle.load(pickle)
		
	elif os.path.isfile(train_csv):
		print "Trained model from {} and save as {}".format(train_csv, model_pkl)
		return model_from_csv(train_csv, features, model_pkl, save_model=False, cv=False)
	
	else:
		print "{} or {} needs to be in the file directory.".format(train_csv, model_pkl)

def string_2_array(raw_in, features, feature_ids):
	# Creates a numerized array with features in the correct indexes. 

	input_dict = {}
	input_dict['Timestamp'] = raw_in.split(':')[0]
	input_dict['Hour'] =  float(raw_in.split(':')[0][-2:] + float(raw_in.split(':')[1])/60)
	input_dict['Browser'] =  raw_in.split(',')[2]
	input_dict['Platform'] =  raw_in.split(',')[3]
	input_dict['Region'] =  raw_in.split(',')[4]
	
	output_array = []
	for feat in features:
		output_array.append(feature_ids[input_dict[feat]])

	return output_array

if __name__ == '__main__':
	
	# Parameters
	train_csv = 'training.csv'
	model_pkl = 'model.pkl'
	output_csv = 'output.csv'
	
	# Choose Features Hour, Browser, Platform, Region. Delete picked model if changed.
	features = ['Hour', 'Region', 'Platform']


	# Load and/or train model, predict best ad via STDIN. 
	model, feature_ids = load_model(train_csv, model_pkl)
	STDIN_run(output_csv, model, feature_ids)
	print 'Done.'


