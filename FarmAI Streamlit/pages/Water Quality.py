import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier



# Read dataset
water = pd.read_csv('water_potability.csv')
water_input = water.drop(columns=["Potability"])
water_target = water["Potability"]

#Splitting the data into data training and data testing
rows = 3276
train = int(0.8*3276)

# 80% data training
water_input_train = water_input[:train]
water_target_train = water_target[:train]

# 20% data testing
water_input_test = water_input[train:]
water_target_test = water_target[train:]

#--------------SVM MODEL TRAINING----------------

#Training and testing linear SVM model
svLinear = SVC(kernel='linear')
svLinear.fit(water_input_train,water_target_train)
linearPred = svLinear.predict(water_input_test)
resultLinear = accuracy_score(water_target_test,linearPred)

  
    
#polynomial svm model
svPolynomial = SVC(kernel='poly')
svPolynomial.fit(water_input_train, water_target_train)
polyPred = svPolynomial.predict(water_input_test)
resultPoly = accuracy_score(water_target_test,polyPred)

# Sigmoid svm model
svSigmoid = SVC(kernel='sigmoid')
svSigmoid.fit(water_input_train,water_target_train)
sigPred = svSigmoid.predict(water_input_test)
resultSig = accuracy_score(water_target_test,sigPred)



#RBF svm model
svRBF= SVC(kernel = 'rbf')
svRBF.fit(water_input_train, water_target_train)
RBFPred = svRBF.predict(water_input_test)
resultRBF = accuracy_score(water_target_test,RBFPred)


#Determine the best kernel for SVM model
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
     
#-----------------------------------------

#-----KNN MODEL TRAINING--------------
knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(water_input_train,water_target_train)
knn5Pred = knn5.predict(water_input_test)
resultKNN5 = accuracy_score(water_target_test,knn5Pred)



knn25 = KNeighborsClassifier(n_neighbors=25)
knn25.fit(water_input_train,water_target_train)
knn25Pred = knn25.predict(water_input_test)
resultKNN25 = accuracy_score(water_target_test,knn25Pred)


knn50 = KNeighborsClassifier(n_neighbors=50)
knn50.fit(water_input_train,water_target_train)
knn50Pred = knn50.predict(water_input_test)
resultKNN50 = accuracy_score(water_target_test,knn50Pred)

knn100 = KNeighborsClassifier(n_neighbors=100)
knn100.fit(water_input_train,water_target_train)
knn100Pred = knn100.predict(water_input_test)
resultKNN100 = accuracy_score(water_target_test,knn100Pred)

resultKNN = [resultKNN5,resultKNN25,resultKNN50,resultKNN100]
maxKNN = max(resultKNN)
knnModel = "" 

if maxKNN ==resultKNN5:
    knnModel = "5 Number of neighbor"
  
elif maxKNN == resultKNN25:
    knnModel = "25 Number of neighbor"
   
elif maxKNN == resultKNN50:
    knnModel = "50 Number of neighbor"
   
elif maxKNN == resultKNN100:
    knnModel = "100 Number of neighbor"
   


#------------------Decision Tree Training------------------------
dt = DecisionTreeClassifier()
dt = dt.fit(water_input_train,water_target_train)
dtpred= dt.predict(water_input_test)
dtResult = accuracy_score(water_target_test, dtpred)




# Start of the webpage
st.header("Water Quality Dataset")
st.write("This dataset describes the quality of the water that is influenced by numerous factors. On top of that, water quality influences the wellbeings of humans in a society.")

#Add sidebar
page = st.sidebar.radio("MORE DETAILS",["Data Input and Data Target", "Training and Testing Dataset"] )


# Display options on sidebar element 
if page == "Data Input and Data Target":
    st.subheader("Data Input and Data Target")
    #Display Data input 
    st.write("Data Input")
    water_input
    st.write("")
    
    #Display Data Target 
    st.write("Data Target")
    water_target
    st.write("")

elif page == "Training and Testing Dataset":
    
    st.subheader("Training And Testing Dataset")
    #Display Training Data
    st.write("Training Dataset")
    water_input_train
    water_target_train

    #Display Testing Dataset
    st.write("Testing Dataset")
    water_input_test
    water_target_test



#Select model and show the result
st.write("")
st.subheader("Select Model:")
st.write("Please select a model to obtain result:")
selectModelWater = st.selectbox("Model Selection", options=["Support Vector Machine","K-Nearest Neighbor","Decision Tree"])
selectInput = st.selectbox("Input Type", options = ["Manual Input","Use Testing Data"])


#Display SVM kernel results
if selectModelWater == "Support Vector Machine":

    st.header("Support Vector Machine")
    st.subheader("SVM Description")
    st.write("For this dataset, the SVM model is used for classifying water potability. The kernels used for this dataset is Linear, Polynomial, Sigmoid and Radial basis Function.")
    st.write("The SVM model classify the objects by calculating the optimal hyperplane.")

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
        dtRs = pd.DataFrame([{
            "Kernel": svmModel, "Accuracy Score": maxSVM
        }])
        st.dataframe(dtRs,hide_index=True, use_container_width=True)
        st.write("The best model is "+svmModel+" with the accuracy score of "+str(maxSVM))


    
    elif selectInput == "Manual Input":
         userInput = st.text_area("Please replace with the input data for prediction ","ph,Hardness,Solids,Chloramines,Sulfate,Conductivitiy,Organic Carbon,Trihalomethanes,Turbidity,Potability")

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
elif  selectModelWater == "K-Nearest Neighbor":

    st.header("K-Nearest Neighbor")
    st.subheader("KNN Description")
    st.write("For this dataset, the KNN model with a few different number of neighbors are implemented. The number of neighbors implemented are 5,25,50,100.")

    if selectInput == "Use Testing Data":
        st.subheader("The Result of KNN Models With Different Neighbors Value")
        st.write("The accuracy score for all KNN models are listed below:")
        dt = pd.DataFrame([
            {"Number of Neighbor": "5", "Accuracy Score":resultKNN5},
            {"Number of Neighbor": "25", "Accuracy Score": resultKNN25},
            {"Number of Neighbor": "50", "Accuracy Score": resultKNN50},
            {"Number of Neighbor": "100", "Accuracy Score": resultKNN100}
                        ])
        
        st.dataframe(dt,use_container_width=True, hide_index=True)

    
    
        # Display best KNN model 
        st.subheader("Choosen KNN Model")
        dtRs = pd.DataFrame([{
            "Number of Neighbor": knnModel, "Accuracy Score": maxKNN
        }])
        st.dataframe(dtRs,hide_index=True, use_container_width=True)
        st.write("The best model is "+knnModel+" with the accuracy score of "+str(maxKNN))

    elif selectInput == "Manual Input":
        userInput = st.text_area("Please replace with the input data for prediction: ","ph,Hardness,Solids,Chloramines,Sulfate,Conductivitiy,Organic Carbon,Trihalomethanes,Turbidity,Potability")
        
        user_data = [float(val.strip()) for val in userInput.split(',') if val.strip()]

        user_data_2D = [user_data]

        #make predictions
        input5 = knn5.predict([user_data_2D][0])
        input25= knn25.predict([user_data_2D][0])
        input50 =knn50.predict([user_data_2D][0])
        input100 = knn100.predict([user_data_2D][0])
        
        st.subheader("The Result From User Input For KNN Model")
        #show the prediction result
        dt = pd.DataFrame([
			{"Number of Neighbor": "5", "Prediction":input5},
			{"Number of Neighbor": "25", "Prediction": input25},
			{"Number of Neighbor": "50", "Prediction": input50},
			{"Number of Neighbor": "100", "Prediction": input100}
						])
		
        st.dataframe(dt, use_container_width=True, hide_index=True)
        



   
#Display DT result
elif selectModelWater == "Decision Tree":
     
    if selectInput == "Use Testing Data":
        st.header("Decision Tree")
        st.subheader("Decision Tree Description")
        st.write( '''A decision tree algorithm is a supervised machine learning algorithm used for both classification and regression tasks. The algorithm creates a tree-like model of decisions based on features present in the training data. It is a predictive modeling tool that recursively splits the dataset into subsets based on the most significant attribute at each node of the tree''' )
        dtRs = pd.DataFrame([{
            "Model": "Decision Tree", "Accuracy Score": dtResult
        }])
        st.dataframe(dtRs,hide_index=True, use_container_width=True)        

        
    elif selectInput == "Manual Input":
        userInput = st.text_area("Please replace with the input data for prediction: ","ph,Hardness,Solids,Chloramines,Sulfate,Conductivitiy,Organic Carbon,Trihalomethanes,Turbidity,Potability")

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
    
   


# <<<<<<STATEFUL BUTTON CONCLUSION RESULT>>>>>>>>
if 'button_state' not in st.session_state:
    st.session_state.button_state = False

clicked = st.button('Click To Compare Models and Find The Best Model')

if clicked:
    st.session_state.button_state = not st.session_state.button_state

if st.session_state.button_state:
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

