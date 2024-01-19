import streamlit as st
import pandas as pd
import numpy as np


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#results_df = pd.DataFrame(columns=['Dataset', 'Model', 'Accuracy'])


st.title("FarmAi: Integrating Machine Learning for Sustainable Farming Solutions")

st.header("This project is about incorporating machine learning techniques in agriculture as it can help farmers by suggesting and predicting various aspects such as suitable fertilizers, suitable pesticides and suitable crop based on various elements such as the soil’s nutrient content, pH level and humidity. ")

st.write("This part is about fertilizer prediction")

selectDataset = st.selectbox("Select Dataset", options = [ "Select Dataset", "Fertilizer Prediction", "Soil Moisture Prediction", "Crop Prediction", "Water Quality Prediction"])

if selectDataset == "Fertilizer Prediction":

	st.subheader("Fertilizer Prediction Full Dataset")
	st.write("Fertilizer prediction dataset is crucial for determining the optimal type and quantity of fertilizers for specific crops, plants and soil conditions. It includes information on soil type, crop type, nitrogen level, and temperature to different fertilizers.")
	dataset = pd.read_csv('Fertilizer Prediction.csv')
	
	dataset

	st.subheader("Data Input For Fertilizer Prediction ")
	datainput = dataset.drop(['Soil Type', 'Crop Type', 'Fertilizer Name'], axis=1)
	
	datainput	

	st.subheader("Data Target For Fertilizer Prediction")
	datatarget = dataset['Fertilizer Name']

	datatarget
	
	skiprowfold1 = [n for n in range (1,19)]
	datainput_training1 = datainput.drop(datainput.index[skiprowfold1])
	datatarget_training1 = datatarget.drop(datatarget.index[skiprowfold1])
	datainput_testing1 = datainput[1:19]
	datatarget_testing1 = datatarget[1:19]	

	skiprowfold2 = [n for n in range (20,40)]
	datainput_training2 = datainput.drop(datainput.index[skiprowfold2])
	datatarget_training2 = datatarget.drop(datatarget.index[skiprowfold2])
	datainput_testing2 = datainput[20:40]
	datatarget_testing2 = datatarget[20:40]

	skiprowfold3 = [n for n in range (40,60)]
	datainput_training3 = datainput.drop(datainput.index[skiprowfold3])
	datatarget_training3 = datatarget.drop(datatarget.index[skiprowfold3])
	datainput_testing3 = datainput[40:60]
	datatarget_testing3 = datatarget[40:60]

	skiprowfold4 = [n for n in range (60,80)]
	datainput_training4 = datainput.drop(datainput.index[skiprowfold4])
	datatarget_training4 = datatarget.drop(datatarget.index[skiprowfold4])
	datainput_testing4 = datainput[60:80]
	datatarget_testing4 = datatarget[60:80]

	skiprowfold5 = [n for n in range (80,99)]
	datainput_training5 = datainput.drop(datainput.index[skiprowfold5])
	datatarget_training5 = datatarget.drop(datatarget.index[skiprowfold5])
	datainput_testing5 = datainput[80:99]
	datatarget_testing5 = datatarget[80:99]
	
	datainput_training = [datainput_training1, datainput_training2, datainput_training3, datainput_training4, datainput_training5]
	datatarget_training = [datatarget_training1, datatarget_training2, datatarget_training3, datatarget_training4, datatarget_training5]
	datainput_testing = [datainput_testing1, datainput_testing2, datainput_testing3, datainput_testing4, datainput_testing5] 
	datatarget_testing =[datatarget_testing1, datatarget_testing2, datatarget_testing3, datatarget_testing4, datatarget_testing5]

	st.subheader("Training and Testing Data Will Be Using 5 Folds Cross-Validation ")

	selectModel = st.selectbox("Select Model", options = ["Select Model", "KNN", "SVM" , "RF"]) 
	
	def model_train(type, n, s):

		accModelAvg = 0
		model_fold = []
		sum_fold = 0

		for i in range(5):

			if type == "RF":
				model = RandomForestClassifier(n_estimators=n)
			elif type == "SVM":
				model = SVC(kernel=s)
			elif type == "KNN":
				model = KNeighborsClassifier(n_neighbors=n)

			model.fit(datainput_training[i], datatarget_training[i])
			modelPredict = model.predict(datainput_testing[i])
			model_as = accuracy_score(modelPredict, datatarget_testing[i])
			model_fold.append(model)
			sum_fold = sum_fold + model_as
	

		accModelAvg = sum_fold/5

		return accModelAvg, model_fold
	
	selectNeighbors = "0"
	selectEstimators = "100"

	if selectModel == "KNN":

		st.subheader("What is KNN ?")
		st.write("KNN is an algorithm is a simple and versatile machine learning approach used for classification and regression tasks. It identifies the k-nearest neighbors of a new data point and assigns a label based on majority voting (for classification) or averages their values. It uses hyperparameter 'k' is as it influencing the algorithm's sensitivity to local patterns. Kernels can be in any but it famous with 1, 5, 10 and 15 kernels.")		
		
		selectNeighbors = st.selectbox("Select Neighbors", options=["Select Neighbors", "10", "20", "30"])
		
		st.write("Predicted Results from Testing Dataset:")
		
		if selectNeighbors != "Select Neighbors":
            		st.write(f"Accuracy Score Using KNN {selectNeighbors}: ")
            		score, model = model_train(selectModel, int(selectNeighbors), "0")
            		st.write(score)

		#results_df = results_df.append({'Dataset': selectDataset, 'Model': 'Random Forest', 'Accuracy': score }, ignore_index = True)

	elif selectModel == "SVM":

		st.subheader("What is SVM ?")
		st.write("SVM is a machine learning algorithm that primarily used for classification and regression task. The difference is it finds a hyperplane in a high-dimensional space that best separates data points into different classes.  SVM can use kernel functions to map data into higher-dimensional spaces such as Linear, RBF, Poly, and Sigmoid.")
		
		selectKernel = st.selectbox("Select Kernel", options=["Select Kernel", "sigmoid", "linear", "poly", "rbf"])

		st.write("Predicted Results from Testing Data:")

		if selectKernel == "sigmoid":

			st.write("Accuracy Score Using SVM Sigmoid: ")
			score, model = model_train(selectModel, 0, selectKernel )
			score

			#results_df = results_df.append({'Dataset': selectDataset, 'Model': 'SVM (Sigmoid)', 'Accuracy':score}, ignore_index=True)

		elif selectKernel == "linear":
			
			st.write("Accuracy Score Using SVM Linear: ")
			score, model = model_train(selectModel, 0, selectKernel )
			score

			#results_df = results_df.append({'Dataset': selectDataset, 'Model': 'SVM (Linear)', 'Accuracy': score}, ignore_index=True)

		elif selectKernel == "poly":
			
			st.write("Accuracy Score Using SVM Poly: ")
			score, model = model_train(selectModel, 0, selectKernel )
			score

			#results_df = results_df.append({'Dataset': selectDataset, 'Model': 'SVM (Poly)', 'Accuracy': score}, ignore_index=True)

		elif selectKernel == "rbf":
			
			st.write("Accuracy Score Using SVM RBF: ")
			score, model = model_train(selectModel, 0, selectKernel )
			score
	
		#results_df = results_df.append({'Dataset': selectDataset, 'Model': 'SVM (RBF)', 'Accuracy': score}, ignore_index=True)

	elif selectModel == "RF":

		st.subheader("What is RF ?")
		st.write("Random Forest is a collection of decision trees, where each tree is trained on a random subset of the training data and makes its own prediction. The final prediction of the Random Forest is often determined by a majority vote (for classification) or an average (for regression) of the individual trees' predictions.")

		selectEstimators = st.selectbox("Select Estimators", options=["Select Estimators", "100", "500", "1000", "1500"])

		st.write("Predicted Results from Testing Data:")

		if selectEstimators != "Select Estimators":
			st.write(f"Accuracy Score Using RF {selectEstimators}: ")

			# Split data into training and testing sets
			datainput_training, datainput_testing, datatarget_training, datatarget_testing = train_test_split(datainput, datatarget, test_size=0.2, random_state=42)

			model = RandomForestClassifier(n_estimators=int(selectEstimators), random_state=42)
			model.fit(datainput_training, datatarget_training)
			modelPredict = model.predict(datainput_testing)
			score = accuracy_score(modelPredict, datatarget_testing)

			st.write(score)

			#results_df = results_df.append({'Dataset': selectDataset, 'Model': 'Random Forest', 'Accuracy': score}, ignore_index=True)

elif selectDataset == "Soil Moisture Prediction":
	st.subheader("Fertilizer Prediction Full Dataset")

elif selectDataset == "Crop Prediction":
	st.subheader("Fertilizer Prediction Full Dataset")

elif selectDataset == "Water Quality Prediction":
	st.subheader("Fertilizer Prediction Full Dataset")

st.header("Conclusion: ")

#st.subheader("Model Performance Comparison Table")
#st.table(results_df)




	