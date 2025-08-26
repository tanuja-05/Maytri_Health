import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import warnings
import pandas as pd
import plotly.express as px
from io import StringIO
import requests
from streamlit_chat import message

from codebase.dashboardgraph import MaternalHealthDashboard

from PIL import Image
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from keras.models import load_model
import requests
from bs4 import BeautifulSoup
import urllib.parse

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

maternal_model = pickle.load(open("Models/random_forest_model.sav",'rb'))
fetal_model = pickle.load(open("Models/fetal_health_classifier.sav",'rb'))
pcos_model = pickle.load(open("Models/pcos_model.sav",'rb'))
pcos_scaler = pickle.load(open("Models/scaler.sav",'rb'))

model = load_model('Models/FV.h5')
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

fruits = ['Apple', 'Banana', 'Bello Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']
vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']

# sidebar for navigation
with st.sidebar:
    st.title("MATRI HEALTH")
    st.write("Welcome to the Matri Health")
    st.write(" Choose an option from the menu below to get started:")

    selected = option_menu('MATRI HEALTH',
                          ['About us',
                          'Pregnancy Risk Prediction',
                          'Fetal Health Prediction',
                          'PCOS Risk Prediction',
                          'Visual analytics',
                          'Food calorie check'],
                          icons=['chat-square-text','hospital','capsule-pill','clipboard-data'],
                          default_index=0)

if (selected == 'About us'):
    
    st.title("Welcome to MATRI HEALTH")
    st.write("At Matri Health, our mission is to revolutionize healthcare by offering innovative solutions through predictive analysis. "
         "Our platform is specifically designed to address the intricate aspects of maternal and fetal health, providing accurate "
         "predictions and proactive risk management.")
    
    col1, col2= st.columns(2)
    with col1:
        # Section 1
        st.header("1. Pregnancy Risk Prediction")
        st.write("Our Pregnancy Risk Prediction feature utilizes advanced algorithms to analyze various parameters, including age, "
                "body sugar levels, blood pressure, and more. By processing this information, we provide accurate predictions of "
                "potential risks during pregnancy.")
     
        st.image("Images/risk_image.jpg", caption="Pregnancy Risk Prediction", use_column_width=True)
    with col2:
        # Section 2
        st.header("2. Fetal Health Prediction")
        st.write("Fetal Health Prediction is a crucial aspect of our system. We leverage cutting-edge technology to assess the "
                "health status of the fetus. Through a comprehensive analysis of factors such as ultrasound data, maternal health, "
                "and genetic factors, we deliver insights into the well-being of the unborn child.")
     
        st.image("Images/health_image.jpg", caption="Fetal Health Prediction", use_column_width=True)

    # Section 3
    st.header("3. Dashboard")
    st.write("Our Dashboard provides a user-friendly interface for monitoring and managing health data. It offers a holistic "
            "view of predictive analyses, allowing healthcare professionals and users to make informed decisions. The Dashboard "
            "is designed for ease of use and accessibility.")
    
    # Closing note
    st.write("Thank you for choosing E-Doctor. We are committed to advancing healthcare through technology and predictive analytics. "
            "Feel free to explore our features and take advantage of the insights we provide.")

if (selected == 'Pregnancy Risk Prediction'):
    
    # page title
    st.title('Pregnancy Risk Prediction')
    content = "Predicting the risk in pregnancy involves analyzing several parameters, including age, blood sugar levels, blood pressure, and other relevant factors. By evaluating these parameters, we can assess potential risks and make informed predictions regarding the pregnancy's health"
    st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div></br>", unsafe_allow_html=True)
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        SistolicBP = st.text_input('SistolicBP in mmHg')
        
    with col2:
        diastolicBP = st.text_input('diastolicBP in mmHg')
    
    with col3:
        BS = st.text_input('Blood glucose in mmol/L')
    
    with col1:
        bodyTemp = st.text_input('Body Temperature ( in Fahrenheit)')

    with col2:
        heartRate = st.text_input('Heart rate in beats per minute')
    
    riskLevel=""
    predicted_risk = [0] 
    # creating a button for Prediction
    with col1:
        if st.button('Predict Pregnancy Risk'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                predicted_risk = maternal_model.predict([[SistolicBP, diastolicBP, BS, bodyTemp, heartRate]])
            # st
            st.subheader("Risk Level:")
            if predicted_risk[0] == 0:
                st.markdown('<bold><p style="font-weight: bold; font-size: 20px; color: green;">Low Risk</p></bold>', unsafe_allow_html=True)
            elif predicted_risk[0] == 1:
                st.markdown('<bold><p style="font-weight: bold; font-size: 20px; color: orange;">Medium Risk</p></Bold>', unsafe_allow_html=True)
            else:
                st.markdown('<bold><p style="font-weight: bold; font-size: 20px; color: red;">High Risk</p><bold>', unsafe_allow_html=True)
    with col2:
        if st.button("Clear"): 
            st.session_state.diastolicBP = 0
            st.session_state.BS = 0
            st.session_state.bodyTemp = 0
            st.session_state.heartRate = 0
            st.experimental_rerun()

if (selected == 'Fetal Health Prediction'):
    
    # page title
    st.title('Fetal Health Prediction')
    
    content = "Cardiotocograms (CTGs) are a simple and cost accessible option to assess fetal health, allowing healthcare professionals to take action in order to prevent child and maternal mortality"
    st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div></br>", unsafe_allow_html=True)
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        BaselineValue = st.text_input('Baseline Value')
        
    with col2:
        Accelerations = st.text_input('Accelerations')
    
    with col3:
        fetal_movement = st.text_input('Fetal Movement')
    
    with col1:
        uterine_contractions = st.text_input('Uterine Contractions')

    with col2:
        light_decelerations = st.text_input('Light Decelerations')
    
    with col3:
        severe_decelerations = st.text_input('Severe Decelerations')

    with col1:
        prolongued_decelerations = st.text_input('Prolongued Decelerations')
        
    with col2:
        abnormal_short_term_variability = st.text_input('Abnormal Short Term Variability')
    
    with col3:
        mean_value_of_short_term_variability = st.text_input('Mean Value Of Short Term Variability')
    
    with col1:
        percentage_of_time_with_abnormal_long_term_variability = st.text_input('Percentage Of Time With ALTV')

    with col2:
        mean_value_of_long_term_variability = st.text_input('Mean Value Long Term Variability')
    
    with col3:
        histogram_width = st.text_input('Histogram Width')

    with col1:
        histogram_min = st.text_input('Histogram Min')
        
    with col2:
        histogram_max = st.text_input('Histogram Max')
    
    with col3:
        histogram_number_of_peaks = st.text_input('Histogram Number Of Peaks')
    
    with col1:
        histogram_number_of_zeroes = st.text_input('Histogram Number Of Zeroes')

    with col2:
        histogram_mode = st.text_input('Histogram Mode')
    
    with col3:
        histogram_mean = st.text_input('Histogram Mean')
    
    with col1:
        histogram_median = st.text_input('Histogram Median')

    with col2:
        histogram_variance = st.text_input('Histogram Variance')
    
    with col3:
        histogram_tendency = st.text_input('Histogram Tendency')
    
    # creating a button for Prediction
    st.markdown('</br>', unsafe_allow_html=True)
    with col1:
        if st.button('Predict Pregnancy Risk'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                predicted_risk = fetal_model.predict([[BaselineValue, Accelerations, fetal_movement,
       uterine_contractions, light_decelerations, severe_decelerations,
       prolongued_decelerations, abnormal_short_term_variability,
       mean_value_of_short_term_variability,
       percentage_of_time_with_abnormal_long_term_variability,
       mean_value_of_long_term_variability, histogram_width,
       histogram_min, histogram_max, histogram_number_of_peaks,
       histogram_number_of_zeroes, histogram_mode, histogram_mean,
       histogram_median, histogram_variance, histogram_tendency]])
            # st.subheader("Risk Level:")
            st.markdown('</br>', unsafe_allow_html=True)
            if predicted_risk[0] == 0:
                st.markdown('<bold><p style="font-weight: bold; font-size: 20px; color: green;">Result  Comes to be  Normal</p></bold>', unsafe_allow_html=True)
            elif predicted_risk[0] == 1:
                st.markdown('<bold><p style="font-weight: bold; font-size: 20px; color: orange;">Result  Comes to be  Suspect</p></Bold>', unsafe_allow_html=True)
            else:
                st.markdown('<bold><p style="font-weight: bold; font-size: 20px; color: red;">Result  Comes to be  Pathological</p><bold>', unsafe_allow_html=True)
    with col2:
        if st.button("Clear"): 
            st.rerun()


if (selected == 'PCOS Risk Prediction'):
    
    # page title
    st.title('PCOS Risk Prediction')
    content = "Predicting the risk in PCOS involves analyzing several parameters, including age, blood sugar levels, blood pressure, and other relevant factors. By evaluating these parameters, we can assess potential risks and make informed predictions regarding the pregnancy's health"
    st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div></br>", unsafe_allow_html=True)
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Age = st.text_input('Age')
        
    with col2:
        BMI = st.text_input('BMI')
    
    with col3:
        Menstrual_Irregularity  = st.text_input('Menstrual_Irregularity')
    
    with col1:
        Testosterone_Level = st.text_input('Testosterone_Level(ng/dL)')

    with col2:
        Antral_Follicle_Count = st.text_input('Antral_Follicle_Count')


    # Prediction button
    with col1:
        if st.button('Predict PCOS Risk'):
            try:
                input_data = [[
                    int(Age), float(BMI), int(Menstrual_Irregularity), float(Testosterone_Level), int(Antral_Follicle_Count)
                ]]
                input_data_scaled = pcos_scaler.transform(input_data)
                prediction = pcos_model.predict(input_data_scaled)
                
                st.subheader("PCOS Risk Level:")
                if prediction[0] == 0:
                    st.markdown('<p style="font-weight: bold; font-size: 20px; color: green;">Low Risk</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p style="font-weight: bold; font-size: 20px; color: red;">High Risk</p>', unsafe_allow_html=True)
            except Exception as e:
                st.error("Error: Please enter valid numerical values.")


if (selected == "Visual analytics"):
    api_key = "579b464db66ec23bdd00000139b0d95a6ee4441c5f37eeae13f3a0b2"
    api_endpoint = api_endpoint= f"https://api.data.gov.in/resource/6d6a373a-4529-43e0-9cff-f39aa8aa5957?api-key={api_key}&format=csv"
    st.header("Visual analytics")
    content = "Our interactive Visual analytics offers a comprehensive visual representation of maternal health achievements across diverse regions. The featured chart provides insights into the performance of each region concerning institutional deliveries compared to their assessed needs. It serves as a dynamic tool for assessing healthcare effectiveness, allowing users to quickly gauge the success of maternal health initiatives."
    st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div></br>", unsafe_allow_html=True)

    dashboard = MaternalHealthDashboard(api_endpoint)
    dashboard.create_bubble_chart()
    with st.expander("Show More"):
    # Display a portion of the data
        content = dashboard.get_bubble_chart_data()
        st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div>", unsafe_allow_html=True)

    dashboard.create_pie_chart()
    with st.expander("Show More"):
    # Display a portion of the data
        content = dashboard.get_pie_graph_data()
        st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div>", unsafe_allow_html=True)



if (selected == "Food calorie check"):
    def fetch_calories(prediction):
        options = Options()
        options.headless = True
        driver = webdriver.Chrome(options=options)

    # Prepare the search query
        query = f"calories in {prediction} per 100g"
        url = f"https://www.bing.com/search?q={query}"

    # Open the Bing search engine URL using Selenium
        driver.get(url)
        cal = "none"
        try:
        # Extract the calorie information using the class name
            calorie_element = driver.find_element(By.CLASS_NAME, "b_focusTextLarge")
            calories = calorie_element.text
            cal = calories
        except Exception as e:
            print("Calories data not found")
        return cal
        driver.quit()
   


    def processed_img(img_path):
        img = load_img(img_path, target_size=(224, 224, 3))
        img = img_to_array(img)
        img = img / 255
        img = np.expand_dims(img, [0])
        answer = model.predict(img)
        y_class = answer.argmax(axis=-1)
        print(y_class)
        y = " ".join(str(x) for x in y_class)
        y = int(y)
        res = labels[y]
        print(res)
        return res.capitalize()


    def run():
        st.title("Food Calorie Detector 🔍")
        img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
        if img_file is not None:
            img = Image.open(img_file).resize((250, 250))
            st.image(img, use_column_width=False)
            save_image_path = './upload_images/' + img_file.name
            with open(save_image_path, "wb") as f:
                f.write(img_file.getbuffer())

        # if st.button("Predict"):
            if img_file is not None:
                result = processed_img(save_image_path)
                print(result)
                if result in vegetables:
                    st.info('**Category : Vegetables**')
                else:
                    st.info('**Category : Fruit**')
                st.success("**Predicted : " + result + '**')
                cal = fetch_calories(result)
                if cal:
                    st.warning('**' + cal + ' ( per 100 grams)**')

    run()
    




# Function to simulate chatbot interaction with close functionality
def chatbot():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_open" not in st.session_state:
        st.session_state.chat_open = True  # Flag to control chat visibility
    
    # Predefined questions and answers
    predefined_questions = {
        "What is Pregnancy Risk Prediction?": "Pregnancy Risk Prediction uses algorithms to predict potential risks during pregnancy by analyzing age, blood pressure, glucose levels, and more.",
        "What is Fetal Health Prediction?": "Fetal Health Prediction assesses the health of the fetus using various parameters like heart rate, fetal movements, and ultrasound data.",
        "About visualization?": "The Dashboard provides a visual representation of maternal health, showcasing charts and analytics about healthcare effectiveness."
    }

    # If the chatbot is open
    if st.session_state.chat_open:
        # Display chatbot interface
        st.title("Chatbot")

        # User input
        user_input = st.text_input("Ask me anything", key="user_input")

        # Add user message to the chat history only if it is a new message
        if user_input and not any(msg["role"] == "user" and msg["text"] == user_input for msg in st.session_state.messages):
            st.session_state.messages.append({"role": "user", "text": user_input})

            # Respond with predefined answers
            if user_input in predefined_questions:
                st.session_state.messages.append({"role": "bot", "text": predefined_questions[user_input]})
            else:
                st.session_state.messages.append({"role": "bot", "text": "Sorry, I don’t understand your question. Please try another one."})

        # Display chat history
        for idx, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                message(msg["text"], is_user=True, key=f"user_{idx}")
            else:
                message(msg["text"], is_user=False, key=f"bot_{idx}")

        # Close chat button
        close_button = st.button("Close Chat", key="close_chat")
        if close_button:
            st.session_state.chat_open = False  # Hide the chat when the button is pressed

    # If the chatbot is closed
    else:
        # Display a floating chat icon
        if st.button("Open Chat", key="open_chat"):
            st.session_state.chat_open = True  # Open the chat when the button is pressed


# Run the chatbot function
chatbot()
