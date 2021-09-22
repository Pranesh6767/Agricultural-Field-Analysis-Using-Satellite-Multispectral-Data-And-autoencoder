# application.py

from flask import Flask
import flask
from flask import request
from flask import render_template
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle


# creates a Flask application, named app
app = Flask(__name__)

def testing(df):
	df.drop(['system:time_start'],axis = 1, inplace = True)
	df = df.astype('str')
	columns = df.columns.tolist()
	for j in columns[:]:
		df[j] = df[j].str.replace(',', '').astype(float)
	nir = df['B5']
	red = df['B4']
	df['ndvi'] =(nir-red)/(nir+red)
	df.sort_values(by=['ndvi'], inplace=True, ascending=False)
	sliced_df = df.iloc[:39,:]
	sliced_df = sliced_df.iloc[:,:-1]
	df_np = np.array(sliced_df)
	df_np = df_np.flatten()
	encoder = load_model('model_v1.h5')
	from tensorflow import keras
	layer_name = 'Feature_latent'
	intermediate_layer_model = keras.Model(inputs=encoder.input,outputs=encoder.get_layer(layer_name).output)
	X_data = df_np/1000
	X_1 = np.expand_dims(X_data, axis=0)
	intermediate_output = intermediate_layer_model(X_1)
	intermediate_output = intermediate_output.numpy()

	filename = 'classifier.sav'
	loaded_model = pickle.load(open(filename, 'rb'))

	input_vector = np.delete(intermediate_output,[3,5])
	input_vector = input_vector.reshape(1, -1)
	output = loaded_model.predict(input_vector)
	output = output[0]
	return output




# a route where we will display a welcome message via an HTML template
@app.route("/", methods = ['GET', 'POST'])
def hello():
	return render_template('index.html')

@app.route("/submit", methods = ['GET', 'POST'])
def submits():
	file_up = request.files['myFile']
	df = pd.read_csv(file_up)
	output = testing(df)
	output = '<h1> Quality Index of Land: '+ str(output) + '</h1>'
	return output

# run the application
if __name__ == "__main__":
    app.run(port = 5000)
