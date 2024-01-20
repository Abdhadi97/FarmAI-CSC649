from sklearn.tree import DecisionTreeClassifier
import streamlit as st
import pandas as pd

# Scikit-Learn
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

description = "The soil fertility dataset is a set of soil samples collection designed for predicting soil fertility based on elemental soil analysis. The dataset encompasses 880 soil samples, each characterized by a set of elemental content and specific features. There are 12 columns in this dataset as the  data input and 1 column as the data target."

inputdatadesc = '''N - Nitrogen (NH4+): The ratio of nitrogen content in the soil.
P - Phosphorous (P): The ratio of phosphorous content in the soil.
K - Potassium (K): The ratio of potassium content in the soil.
pH - Soil Acidity: A measure of soil acidity.
EC - Electrical Conductivity: An indicator of the soil's ability to conduct electricity.
OC - Organic Carbon: The proportion of organic carbon in the soil.
S - Sulfur (S): The sulfur content in the soil.
Zn - Zinc (Zn): The zinc content in the soil.
Fe - Iron (Fe): The iron content in the soil.
Cu - Copper (Cu): The copper content in the soil.
Mn - Manganese (Mn): The manganese content in the soil.
B - Boron (B): The boron content in the soil.'''

targetdatadesc = '''Class 0 ("Less Fertile"): Indicates soil with lower fertility levels.
Class 1 ("Fertile"): Represents soil with moderate fertility, suitable for general cultivation.
Class 2 ("Highly Fertile"): Denotes soil with high fertility, ideal for robust plant growth and productivity.'''

# Read the dataset
soil = pd.read_csv('soil-fertility.csv')
X_soil = soil.drop('Output', axis = 1)
y_soil = soil['Output']


#Spliting Dataset
X_train, X_test, y_train, y_test = train_test_split(X_soil, y_soil, test_size=0.2, random_state=42)

#<<<<<<<<<<<<<<<<SVM MODEL TRAINING>>>>>>>>>>>>>>>

svLinear = SVC(kernel='linear')
svPolynomial = SVC(kernel='poly')
svSigmoid = SVC(kernel='sigmoid')
svRBF= SVC(kernel = 'rbf')
svLinear.fit(X_train,y_train)
svPolynomial.fit(X_train,y_train)
svSigmoid.fit(X_train,y_train)
svRBF.fit(X_train,y_train)
linearPred = svLinear.predict(X_test)
polyPred = svPolynomial.predict(X_test)
sigmoidPred = svSigmoid.predict(X_test)
rbfPred = svRBF.predict(X_test)

resultLinear = accuracy_score(y_test,linearPred)
resultPoly = accuracy_score(y_test,polyPred)
resultSig = accuracy_score(y_test,sigmoidPred)
resultRBF = accuracy_score(y_test,rbfPred)

#The best kernel
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


#<<<<<<<<<<<<<<<<<<<<<<<<<<<< KNN MODEL TRAINING >>>>>>>>>>>>>>>>>>>>>>>>>
knn10 = KNeighborsClassifier(n_neighbors=10)
knn20 = KNeighborsClassifier(n_neighbors=20)
knn30 = KNeighborsClassifier(n_neighbors=30)
knn40 = KNeighborsClassifier(n_neighbors=40)
knn10.fit(X_train, y_train)
knn20.fit(X_train, y_train)
knn30.fit(X_train, y_train)
knn40.fit(X_train, y_train)
knn10Pred = knn10.predict(X_test)
knn20Pred = knn20.predict(X_test)
knn30Pred = knn30.predict(X_test)
knn40Pred = knn40.predict(X_test)
resultKNN10= accuracy_score(y_test,knn10Pred)
resultKNN20= accuracy_score(y_test,knn20Pred)
resultKNN30= accuracy_score(y_test,knn30Pred)
resultKNN40= accuracy_score(y_test,knn40Pred)

resultKNN = [resultKNN10,resultKNN20,resultKNN30,resultKNN40]
maxKNN = max(resultKNN)
knnModel = "" 
if maxKNN ==resultKNN10:
    knnModel = "10 Number of neighbor"
  
elif maxKNN == resultKNN20:
    knnModel = "20 Number of neighbor"
   
elif maxKNN == resultKNN30:
    knnModel = "30 Number of neighbor"
   
elif maxKNN == resultKNN40:
    knnModel = "40 Number of neighbor"

#<<<<<<<<<<<<<<DECISION TREE TRAINING>>>>>>>>>>>>>>

dt = DecisionTreeClassifier()
dt = dt.fit(X_train,y_train)
dtpred= dt.predict(X_test)
dtResult= accuracy_score(y_test,dtpred)



#<<<<<<<<<<<<<<<<Start of the webpage>>>>>>>>>>>>>>>>>>>>>>
st.title("SOIL FERTILITY DATASET")
st.markdown(description)

# Add a sidebar
page = st.sidebar.radio("MORE DETAILS", ["Data Input and Data Target", "Training and Testing Dataset"])


# Display content based on sidebar choice
if page == "Data Input and Data Target":
    # Data Input
    st.subheader("Data Input")
    X_soil
    st.text(inputdatadesc)

    # Data Target
    st.subheader("Data Target")
    y_soil
    st.text(targetdatadesc)

elif page == "Training and Testing Dataset":
    #Training Data
    st.subheader("Training Data")
    X_train
    y_train

    st.subheader("Testing Data")
    X_test
    y_test

# select model elements
SelectModelSoil = st.selectbox("ALGORITHM",["Choose model","SVM","KNN","DT"])

# Display content based on model choice
if SelectModelSoil == "SVM":
    st.subheader("Support Vector Machines (SVM)")
    
    # Provide information about SVM
    st.write('''Support Vector Machines (SVM) is a powerful algorithm for classification tasks.
             It works well for both linear and non-linear datasets by finding the optimal hyperplane.''')

    # Input elements for SVM kernel
    st.subheader("The Result of SVM Models With Different Kernels")
    st.write("The accuracy score for all SVM kernel models are listed below:")
    dtKNN = pd.DataFrame([
        {"Kernel": "Linear", "Accuracy Score": resultLinear},
        {"Kernel": "Polynomial", "Accuracy Score": resultPoly},
        {"Kernel": "RBF", "Accuracy Score": resultRBF},
        {"Kernel": "Sigmoid", "Accuracy Score": resultSig}
    ])
    st.dataframe(dtKNN, use_container_width=True, hide_index=True)
   
    # Display the best svm kernel
    st.subheader("Choosen SVM Kernel")
    st.write("The best SVM kernel with the highest accuracy score: ")
    st.write(svmModel)
    st.write("With the accuracy score of " + str(maxSVM))
    
 

elif SelectModelSoil == "KNN":
    st.subheader("K-Nearest Neighbors (KNN)")

    # Provide information about KNN
    st.write('''K-Nearest Neighbors (KNN) is a simple and effective algorithm for classification tasks.
             It classifies a data point based on the majority class of its k-nearest neighbors.''')
    
    st.subheader("The Result of KNN Models With Different Neighbors Value")
    st.write("The accuracy score for all KNN models are listed below:")
    dt = pd.DataFrame([
        {"Number of Neighbor": "10", "Accuracy Score":resultKNN10},
        {"Number of Neighbor": "20", "Accuracy Score": resultKNN20},
        {"Number of Neighbor": "30", "Accuracy Score": resultKNN30},
        {"Number of Neighbor": "40", "Accuracy Score": resultKNN40}
                    ])
	
    st.dataframe(dt,use_container_width=True, hide_index=True)

 
 
    # Display best KNN model 
    st.subheader("Choosen KNN Model")
    st.write("The best KNN model with the highest accuracy score: "+str(knnModel))

    st.write("With the accuracy score of " + str(maxKNN))


elif SelectModelSoil == "DT":
    st.subheader("Decision Tree (DT)")

    # Provide information about Random Forest
    st.write('''Random Forest is an ensemble learning method that builds multiple decision trees and merges them to improve accuracy.
             It's robust, handles non-linear relationships well, and helps prevent overfitting''')
     
    st.write("The accuracy score of decision tree:")
    st.write(dtResult)
    
#Compare Results Between Models
st.subheader("List Of All Models Accuracy Score")

ModelTable = pd.DataFrame(
    [
        {"Model": "SVM","Accuracy Score":maxSVM},
        {"Model": "KNN","Accuracy Score":maxKNN},
        {"Model": "Decision Tree","Accuracy Score": dtResult}
    ]  
)
st.dataframe(ModelTable,use_container_width=True, hide_index=True)

st.subheader("The Best Model")
st.write("The best model between SVM, KNN, and Decision Tree is:")

#Compare Results of three models scores
arrayModel = [maxSVM,maxKNN,dtResult]
AllmodelResult = max(arrayModel)

if AllmodelResult == maxSVM:
     st.write("The SVM model specifically the "+str(svmModel)+" kernel is the best model between the three models with the accuracy score of "+str(maxSVM))
elif AllmodelResult == maxKNN:
     st.write("The KNN model specifically the "+str(knnModel)+" model is the best model between three models with the accuracy score of " +str(maxKNN))
elif AllmodelResult == dtResult:
     st.write("The Decision Tree model is the best model between the three models with the accuracy value of "+str(dtResult))




    

    

