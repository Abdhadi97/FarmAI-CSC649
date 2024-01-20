import streamlit as st
import webbrowser

dataset_link = {
    "Crop Recommendation": "https://www.kaggle.com/datasets/siddharthss/crop-recommendation-dataset",
    "Fertilizer Prediction": "https://www.kaggle.com/code/karthikreddy77/fertilizer-prediction",
    "Water Quality Prediction": "https://www.kaggle.com/datasets/adityakadiwal/water-potability",
    "Soil Fertility": "https://www.kaggle.com/code/nouraalgohary/soil-fertility-prediction"
}

st.set_page_config(
    page_title="Agri-Tech",
    page_icon="🐳",
)

st.sidebar.markdown("## Kaggle-dataset link")
for link_text, link_url in dataset_link.items():
    if st.sidebar.button(link_text):
        webbrowser.open(link_url)
    
        
st.title("FarmAI: INTEGRATING MACHINE LEARNING FOR SUSTAINABLE FARMING SOLUTIONS")
st.text_area("Description of this project", '''FarmAI is a trained machine learning model that focusing on agriculture management. This project is about incorporating machine learning techniques in agriculture as it can help farmers by suggesting and predicting various aspects such as suitable fertilizers, suitable pesticides and suitable crop based on various elements such as the soil’s nutrient content, pH level and humidity.
             ''')
select = dataset = st.selectbox("Select Dataset", options = ["Select Option", "Crop Recommendation", "Fertilizer Prediction", "Water Quality Prediction", "Soil Fertility"])

