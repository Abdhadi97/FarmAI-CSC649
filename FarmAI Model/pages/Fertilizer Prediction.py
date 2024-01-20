from sklearn.tree import DecisionTreeClassifier
import streamlit as st
import pandas as pd
import numpy as np


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

results_df = pd.DataFrame(columns=['Dataset', 'Model', 'Accuracy'])


st.title("Fertilizer Prediction Dataset")

#st.header("This project is about incorporating machine learning techniques in agriculture as it can help farmers by suggesting and predicting various aspects such as suitable fertilizers, suitable pesticides and suitable crop based on various elements such as the soil’s nutrient content, pH level and humidity. ")

#st.write("This part is about fertilizer prediction")


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

sumKNN10 = 0
sumKNN20 = 0
sumKNN30 = 0
sumKNN40 = 0

#KNN training

for i in range(5):
	knn10model = KNeighborsClassifier(n_neighbors=10)
	knn20model = KNeighborsClassifier(n_neighbors=20)
	knn30model = KNeighborsClassifier(n_neighbors=30)
	knn40model = KNeighborsClassifier(n_neighbors=40)
	knn10model.fit(datainput_training[i], datatarget_training[i])
	knn20model.fit(datainput_training[i], datatarget_training[i])
	knn30model.fit(datainput_training[i], datatarget_training[i])
	knn40model.fit(datainput_training[i], datatarget_training[i])
	knn10pred = knn10model.predict(datainput_testing[i])
	knn20pred = knn20model.predict(datainput_testing[i])
	knn30pred = knn30model.predict(datainput_testing[i])
	knn40pred = knn40model.predict(datainput_testing[i])
	ASknn10 = accuracy_score(knn10pred,datatarget_testing[i])
	ASknn20 = accuracy_score(knn20pred,datatarget_testing[i])
	ASknn30 = accuracy_score(knn30pred,datatarget_testing[i])
	ASknn40 = accuracy_score(knn40pred,datatarget_testing[i])
	sumKNN10 = sumKNN10 + ASknn10
	sumKNN20 = sumKNN20 + ASknn20
	sumKNN30 = sumKNN30 + ASknn30
	sumKNN40 = sumKNN40 + ASknn40



AvgKNN10 = sumKNN10/5
AvgKNN20 = sumKNN20/5
AvgKNN30 = sumKNN30/5
AvgKNN40 = sumKNN40/5

#Deetermine the best KNN model
resultKNN = [AvgKNN10,AvgKNN20,AvgKNN30,AvgKNN40]
maxKNN = max(resultKNN)
knnModel = "" 

if maxKNN ==AvgKNN10:
    knnModel = "Number of Neighbor 10"
  
elif maxKNN == AvgKNN20:
    knnModel = "Number of Neighbor 20"
   
elif maxKNN == AvgKNN30:
    knnModel = "Number of Neighbor 30"
elif maxKNN == AvgKNN40:
    KnnModel = "Number of Neighbor 40"
   

#SVM model training
for i in range(5):
	svLinear = SVC(kernel='linear')
	svPolynomial = SVC(kernel='poly')
	svSigmoid = SVC(kernel='sigmoid')
	svRBF= SVC(kernel = 'rbf')
	svLinear.fit(datainput_training[i], datatarget_training[i])
	svPolynomial.fit(datainput_training[i], datatarget_training[i])
	svRBF.fit(datainput_training[i], datatarget_training[i])
	svSigmoid.fit(datainput_training[i], datatarget_training[i])
	predLinear= svLinear.predict(datainput_testing[i])
	predPoly = svPolynomial.predict(datainput_testing[i])
	predSig = svSigmoid.predict(datainput_testing[i])
	predRBF = svRBF.predict(datainput_testing[i])
	resultLinear =  accuracy_score(predLinear, datatarget_testing[i])
	resultPoly= accuracy_score(predPoly, datatarget_testing[i])
	resultRBF = accuracy_score(predRBF, datatarget_testing[i])
	resultSig = accuracy_score(predLinear, datatarget_testing[i])
      
#Determine the best  kernel for SVM
resultSVM = [resultLinear,resultPoly,resultRBF,resultSig]
maxSVM = max(resultSVM)
svmModel = ""

if maxSVM == resultLinear:
        svmModel = "Linear"
      
elif maxSVM == resultPoly:
        svmModel="Polynomial"
      
elif maxSVM == resultSig:
        svmModel = "Sigmoid"
       
elif maxSVM == resultRBF:
        svmModel = "RBF kernel"
        
#Decision Tree Training
for i in range(5):
    dt = DecisionTreeClassifier()
    dt = dt.fit(datainput_training[i], datatarget_training[i])
    dtpred = dt.predict(datainput_testing[i])
    resultDT = accuracy_score(dtpred, datatarget_testing[i])



selectModel = st.selectbox("Select Model", options = ["Select Model", "KNN", "SVM" , "DT"]) 


if selectModel == "KNN":
	st.subheader("What is KNN?")
	st.write("KNN is an algorithm is a simple and versatile machine learning approach used for classification and regression tasks. It identifies the k-nearest neighbors of a new data point and assigns a label based on majority voting (for classification) or averages their values. It uses hyperparameter 'k' is as it influencing the algorithm's sensitivity to local patterns. Kernels can be in any but it famous with 1, 5, 10 and 15 kernels.")

	st.subheader("Results from KNN models: ")
	dt = pd.DataFrame([
		{"Number of Neighbor": "10", "Accuracy Score":AvgKNN10},
		{"Number of Neighbor": "20", "Accuracy Score": AvgKNN20},
		{"Number of Neighbor": "30", "Accuracy Score": AvgKNN30},
        {"Number of Neighbor": "40", "Accuracy Score": AvgKNN40}
					])
	
	st.dataframe(dt,use_container_width=True, hide_index=True)
	st.subheader("Choosen KNN model")
	st.write("Based on the table above, the best KNN model for this dataset is" + str(knnModel))
	st.write("with the accuracy score of " +str(maxKNN))

elif selectModel == "SVM":
    st.subheader("What is SVM")
    st.write("SVM is a machine learning algorithm that is primarily used for classification and regression tasks. The difference is it finds a hyperplane in a high-dimensional space that best separates data points into different classes. SVM can use kernel functions to map data into higher-dimensional spaces such as Linear, RBF, Poly, and Sigmoid.")
    st.subheader("Results from SVM kernels: ")
    
    dtKNN = pd.DataFrame([
        {"Kernel": "Linear", "Accuracy Score": resultLinear},
        {"Kernel": "Polynomial", "Accuracy Score": resultPoly},
        {"Kernel": "RBF", "Accuracy Score": resultRBF},
        {"Kernel": "Sigmoid", "Accuracy Score": resultSig}
    ])
    st.dataframe(dtKNN, use_container_width=True, hide_index=True)
    st.subheader("Choosen SVM model")
    st.write("Based on the table above, the best SVM model for this dataset is " + str(svmModel))
    st.write("with the accuracy score of " +str(maxSVM))

elif selectModel == "DT":
    
	st.subheader("What is Decision Tree")
	st.write("Decison Tree works by recursively partitioning the data into subsets based on the values of input features, creating a tree-like structure where each node represents a decision based on a specific feature.")
	st.subheader("Result of Decision Tree model:")
	dtDT = pd.DataFrame([
			{"Model":"Decision Tree","Accuracy Score":resultDT}
		])	
	st.dataframe(dtDT, use_container_width=True, hide_index=True)
	st.write("Based on the table above, the decision tree model has an accuracy of "+str(resultDT))
	







