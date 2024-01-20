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

# Start of the webpage
st.title("SOIL FERTILITY DATASET")
st.markdown(description)

# Add a sidebar
page = st.sidebar.radio("MORE DETAILS", ["Data Input and Data Target", "Training and Testing Dataset"])

# Additional sidebar elements
model = st.sidebar.selectbox("ALGORITHM",["Choose model","SVM","KNN","DT"])

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

# Display content based on model choice
if model == "SVM":
    st.subheader("Support Vector Machines (SVM)")
    
    # Provide information about SVM
    st.write('''Support Vector Machines (SVM) is a powerful algorithm for classification tasks.
             It works well for both linear and non-linear datasets by finding the optimal hyperplane.''')

    # Input elements for SVM kernel
    kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
    svm_model = SVC(kernel=kernel)
    svm_model.fit(X_train, y_train)
    accuracy = svm_model.score(X_test, y_test)
    st.write(f"Model Accuracy: {accuracy:.2%}")

elif model == "KNN":
    st.subheader("K-Nearest Neighbors (KNN)")

    # Provide information about KNN
    st.write('''K-Nearest Neighbors (KNN) is a simple and effective algorithm for classification tasks.
             It classifies a data point based on the majority class of its k-nearest neighbors.''')
    
    # Input elements for KNN parameters
    n_neighbors = st.slider("Number of Neighbors (k)", 1, 80, 20)
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_train, y_train)
    accuracy = knn_model.score(X_test, y_test)
    st.write(f"Model Accuracy: {accuracy:.2%}")

elif model == "DT":
    st.subheader("Decision Tree (DT)")

    # Provide information about Random Forest
    st.write('''Random Forest is an ensemble learning method that builds multiple decision trees and merges them to improve accuracy.
             It's robust, handles non-linear relationships well, and helps prevent overfitting''')
    
    # Input elements for Random Forest parameters
    
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    accuracy = dt_model.score(X_test, y_test)
    st.write(f"Model Accuracy: {accuracy:.2%}")

    

    

