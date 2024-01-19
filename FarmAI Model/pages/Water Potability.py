import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

st.header("Water Quality Dataset")
st.write("This dataset describes the quality of the water that is influenced by numerous factors. On top of that, water quality influences the wellbeings of humans in a society.")

st.subheader("Data Input and Data Target")
st.write("Below is the features that were used for data input and data target")

st.write("")
st.write("Data Input")
water = pd.read_csv('water_potability.csv')
water_input = water.drop(columns=["Potability"])
water_input

st.write("Data Target")
water_target = water["Potability"]
water_target

#Splitting the data into data training and data testing
rows = 3276
train = int(0.8*3276)

# 80% data training
water_input_train = water_input[:train]
water_target_train = water_target[:train]

# 20% data testing
water_input_test = water_input[train:]
water_target_test = water_target[train:]


st.write("")
st.subheader("Select Model:")
st.write("Please select a model to obtain result:")
selectModelWater = st.selectbox("Model Selection", options=["Support Vector Machine","K-Nearest Neighbor","Decision Tree"])

if selectModelWater == "Support Vector Machine":

    st.header("Support Vector Machine")
    st.subheader("SVM Description")
    st.write("For this dataset, the SVM model is used for classifying water potability. The kernels used for this dataset is Linear, Polynomial, Sigmoid and Radial basis Function.")
    
    st.subheader("Accuracy Score")
    st.write("Linear Kernel")

    #Training and testing linear SVM model
    svLinear = SVC(kernel='linear')
    svLinear.fit(water_input_train,water_target_train)
    linearPred = svLinear.predict(water_input_test)
    result = accuracy_score(water_target_test,linearPred)
    st.write(accuracy_score(water_target_test,linearPred)) 
  
    
    st.write("Polynomial kernel")
    #polynomial svm model
    svPolynomial = SVC(kernel='poly')
    svPolynomial.fit(water_input_train, water_target_train)
    polyPred = svPolynomial.predict(water_input_test)
    st.write(accuracy_score(water_target_test,polyPred)) 


    st.write("Sigmoid kernel")
    # Sigmoid svm model
    svSigmoid = SVC(kernel='sigmoid')
    svSigmoid.fit(water_input_train,water_target_train)
    sigPred = svSigmoid.predict(water_input_test)
    st.write(accuracy_score(water_target_test,sigPred)) 

    st.write("RBF kernel")
    #RBF svm model
    svRBF= SVC(kernel = 'rbf')
    svRBF.fit(water_input_train, water_target_train)
    RBFPred = svRBF.predict(water_input_test)
    st.write(accuracy_score(water_target_test,RBFPred)) 
    
elif  selectModelWater == "K-Nearest Neighbor":

    st.header("K-Nearest Neighbor")
    st.subheader("KNN Description")
    st.write("For this dataset, the KNN model with a few different number of neighbors are implemented. The number of neighbors implemented are 5,25,50,100.")

    st.subheader("Accuracy Score")
    st.write("The accuracy score for all KNN models are listed below:")
    st.write("5 Number of Neighbors")

    knn5 = KNeighborsClassifier(n_neighbors=5)
    knn5.fit(water_input_train,water_target_train)
    knn5Pred = knn5.predict(water_input_test)
    st.write(accuracy_score(water_target_test,knn5Pred))


    st.write("25 Number of Neighbors")

    knn25 = KNeighborsClassifier(n_neighbors=25)
    knn25.fit(water_input_train,water_target_train)
    knn25Pred = knn25.predict(water_input_test)
    st.write(accuracy_score(water_target_test,knn25Pred))

    st.write("50 Number of Neighbors")

    knn50 = KNeighborsClassifier(n_neighbors=50)
    knn50.fit(water_input_train,water_target_train)
    knn50Pred = knn50.predict(water_input_test)
    st.write(accuracy_score(water_target_test,knn50Pred))

    st.write("100 Number of Neighbors")

    knn100 = KNeighborsClassifier(n_neighbors=100)
    knn100.fit(water_input_train,water_target_train)
    knn100Pred = knn100.predict(water_input_test)
    st.write(accuracy_score(water_target_test,knn100Pred))


elif selectModelWater == "Decision Tree":
     
     st.header("Decision Tree")
     st.subheader("Decision Tree Description")
     st.write("For this dataset, the decision tree ")
     
     dt = DecisionTreeClassifier()
     dt = dt.fit(water_input_train,water_target_train)
     dtpred= dt.predict(water_input_test)
     st.write(accuracy_score(water_target_test, dtpred))
    


