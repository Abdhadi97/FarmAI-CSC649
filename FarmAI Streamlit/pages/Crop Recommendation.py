import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

#-------------------------MODEL TRAINING--------------------------
#Read dataset 
crop_train=pd.read_csv("Crop_recommendation_train.csv")
crop_test = pd.read_csv("Crop_recommendation_test.csv")

#x input, y target
x_train = crop_train.drop('label',axis=1)
y_train = crop_train['label']

x_test = crop_test.drop('label',axis=1)
y_test = crop_test['label']


#<<<<<<<SVM MODEL TRAINING>>>>>>>>>>>>>>>>>>>>>>>>

svLinear = SVC(kernel='linear')
svPolynomial = SVC(kernel='poly')
svSigmoid = SVC(kernel='sigmoid')
svRBF= SVC(kernel = 'rbf')
svLinear.fit(x_train,y_train)
svPolynomial.fit(x_train,y_train)
svSigmoid.fit(x_train,y_train)
svRBF.fit(x_train,y_train)
linearPred = svLinear.predict(x_test)
polyPred = svPolynomial.predict(x_test)
sigmoidPred = svSigmoid.predict(x_test)
rbfPred = svRBF.predict(x_test)

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


#<<<<<<<<<<<KNN MODEL TRAINING>>>>>>>>>>>>>>>>>
knn10 = KNeighborsClassifier(n_neighbors=10)
knn20 = KNeighborsClassifier(n_neighbors=20)
knn30 = KNeighborsClassifier(n_neighbors=30)
knn40 = KNeighborsClassifier(n_neighbors=40)
knn10.fit(x_train, y_train)
knn20.fit(x_train, y_train)
knn30.fit(x_train, y_train)
knn40.fit(x_train, y_train)
knn10Pred = knn10.predict(x_test)
knn20Pred = knn20.predict(x_test)
knn30Pred = knn30.predict(x_test)
knn40Pred = knn40.predict(x_test)
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
dt = dt.fit(x_train,y_train)
dtpred= dt.predict(x_test)
dtResult= accuracy_score(y_test,dtpred)


#<<<<<<<<<<<<<<<<<<<<<WEBPAGE STARTS>>>>>>>>>>>>>>>>>>>
st.header("Crop Recommendation Dataset")
st.write("""Crop recommendation dataset comprises of 2200 samples and 8 columns. 
         Each sample represents the type of the crop that most suitable to grow within various parameters.
         There are 7 parameters which are ratio of 
         Nitrogen content in soil (N), ratio of Phosphorous content in soil (P), ratio of Potassium in soil (K), temperature, humidity, pH value of the soil (ph) and rainfail""")

#Add sidebar
page = st.sidebar.radio("MORE DETAILS",["Data Input and Data Target", "Training and Testing Dataset"] )

# Display options on sidebar element 
if page == "Data Input and Data Target":
    st.subheader("Data Input and Data Target")
    #Display Data input 
    st.write("Data Input")
    crop_train
    st.write("")
    
    #Display Data Target 
    st.write("Data Target")
    crop_test
    st.write("")

elif page == "Training and Testing Dataset":
    
    st.subheader("Training And Testing Dataset")
    #Display Training Data
    st.write("Training Dataset")
    x_train
    y_train
    #Display Testing Dataset
    st.write("Testing Dataset")
    x_test
    y_test


#Select model and show the result
st.write("")
st.subheader("Select Model:")
st.write("Please select a model to obtain result:")
selectModelCrop = st.selectbox("Model Selection", options=["Support Vector Machine","K-Nearest Neighbor","Decision Tree"])
selectInput = st.selectbox("Input Type", options = ["Manual Input","Use Testing Data"])

#Display SVM kernel results
if selectModelCrop == "Support Vector Machine":

    st.header("Support Vector Machine")
    st.subheader("SVM Description")
    st.write("For this dataset, the SVM model is used for recommending the suitable crop based on the various parameters of the soil's condition . The kernels used for this dataset is Linear, Polynomial, Sigmoid and Radial basis Function.")
    st.write("The SVM model classify the objects by calculating the optimal hyperplane in the feature space.")
    
    if selectInput == "Use Testing Data":
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
        dtRs = pd.DataFrame([
             {"Kernel": svmModel, "Accuracy Score": maxSVM}
        ])
        st.dataframe(dtRs, use_container_width=True, hide_index=True)
        st.write("The best model is "+svmModel+" with the accuracy score of "+str(maxSVM))

    else:
         userInput = st.text_area("Please replace with the input data for prediction ","N,P,K,temperature,humidity,ph,rainfall")

         user_data = [float(val.strip()) for val in userInput.split(',') if val.strip()]

         user_data_2D = [user_data]

         #make predictions
         inputLinear = svLinear.predict([user_data_2D][0])
         inputPoly= svPolynomial.predict([user_data_2D][0])
         inputRBF =svRBF.predict([user_data_2D][0])
         inputSig = svSigmoid.predict([user_data_2D][0])
         
         st.subheader("The Result From User Input For SVM Model")
         #show the prediction result
         dtKNN = pd.DataFrame([
            {"Kernel": "Linear", "Prediction": inputLinear},
            {"Kernel": "Polynomial", "Prediction": inputPoly},
            {"Kernel": "RBF", "Prediction": inputRBF},
            {"Kernel": "Sigmoid", "Prediction": inputSig}
         ])
         st.dataframe(dtKNN, use_container_width=True, hide_index=True)



#Display KNN kernels results
elif  selectModelCrop == "K-Nearest Neighbor":

   


    st.header("K-Nearest Neighbor")
    st.subheader("KNN Description")
    st.write("For this dataset, the KNN model with a few different number of neighbors are implemented. The number of neighbors implemented are 10,20,30,40.")
    if selectInput == "Use Testing Data":
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
        dtRs = pd.DataFrame([
            {"Number of Neighbor": knnModel, "Accuracy Score":maxKNN}
        ])
        st.dataframe(dtRs, use_container_width=True, hide_index=True)
        st.write("The best model is "+knnModel+" with the accuracy score of "+str(maxKNN))
  

    else:
        userInput = st.text_area("Please replace with the input data for prediction: ","N,P,K,temperature,humidity,ph,rainfall")

        user_data = [float(val.strip()) for val in userInput.split(',') if val.strip()]

        user_data_2D = [user_data]

        #make predictions
        input10 = knn10.predict([user_data_2D][0])
        input20= knn20.predict([user_data_2D][0])
        input30 =knn30.predict([user_data_2D][0])
        input40 = knn40.predict([user_data_2D][0])
        
        st.subheader("The Result From User Input For KNN Model")
        #show the prediction result
        dt = pd.DataFrame([
			{"Number of Neighbor": "10", "Prediction":input10},
			{"Number of Neighbor": "20", "Prediction": input20},
			{"Number of Neighbor": "30", "Prediction": input30},
			{"Number of Neighbor": "40", "Prediction": input40}
						])
		
        st.dataframe(dt, use_container_width=True, hide_index=True)




   
#Display DT result
elif selectModelCrop == "Decision Tree":
     
    if selectInput == "Use Testing Data":
        st.header("Decision Tree")
        st.subheader("Decision Tree Description")
        st.write( '''A decision tree algorithm is a supervised machine learning algorithm used for both classification and regression tasks. The algorithm creates a tree-like model of decisions based on features present in the training data. It is a predictive modeling tool that recursively splits the dataset into subsets based on the most significant attribute at each node of the tree''' )

        
        
        st.write("The accuracy score of decision tree:" + str(dtResult))
        
        
    elif selectInput == "Manual Input":
        userInput = st.text_area("Please replace with the input data for prediction: ","N,P,K,temperature,humidity,ph,rainfall")

        user_data = [float(val.strip()) for val in userInput.split(',') if val.strip()]

        user_data_2D = [user_data]

        #make predictions
        dtPred = dt.predict([user_data_2D][0])

        
        st.subheader("The Result From User Input For Decision Tree Model")
        #show the prediction result
        dtKNN = pd.DataFrame([
        {"Model": "Decision Tree", "Prediction": dtPred},
        ])
        st.dataframe(dtKNN, use_container_width=True, hide_index=True)
    
# <<<<<<BUTTON CONCLUSION RESULT>>>>>>>>
if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True

st.button('Click To Compare Models and Find the Best Model', on_click=click_button)

if st.session_state.clicked:
    # Compare Results Between Models
    st.subheader("List Of All Models Accuracy Score")

    ModelTable = pd.DataFrame(
        [
            {"Model": "SVM", "Accuracy Score": maxSVM},
            {"Model": "KNN", "Accuracy Score": maxKNN},
            {"Model": "Decision Tree", "Accuracy Score": dtResult}
        ]
    )
    st.dataframe(ModelTable, use_container_width=True, hide_index=True)

    st.subheader("The Best Model")
    st.write("The best model between SVM, KNN, and Decision Tree is:")

    # Compare Results of three models scores
    arrayModel = [maxSVM, maxKNN, dtResult]
    AllmodelResult = max(arrayModel)

    if AllmodelResult == maxSVM:
        st.write("The SVM model specifically the " + str(svmModel) + " kernel is the best model between the three models with the accuracy score of " + str(maxSVM))
    elif AllmodelResult == maxKNN:
        st.write("The KNN model specifically the " + str(knnModel) + " model is the best model between three models with the accuracy score of " + str(maxKNN))
    elif AllmodelResult == dtResult:
        st.write("The Decision Tree model is the best model between the three models with the accuracy value of " + str(dtResult))
