import streamlit as st
import tensorflow as tf
import numpy as np
#tensoflow model prection 
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])#convert single image to batch
    prediction = model.predict(input_arr)
    return np.argmax(prediction)# retrun index of max element
#sidebar 
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Model Prediction"])

#Main Page
if(app_mode=="Home"):
    st.header("FURITS & VEGTABLES RECOGNITION SYSTEM")
    image_path="home_img.jpg"
    st.image(image_path)

#About Project
elif(app_mode=="About Project"):
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This dataset images of the following food items:")
    st.code("fruits banana, apple, pear, grapes, orange, kiwi")
    st.code("watermelon, pomegranate, pineapple, mango.vegetables cucumber, carrot")
    st.code("capsicum, onion, potato, lemon, tomato, raddish, beetroot")
    st.code("cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper")
    st.code("chilli pepper, turnip, corn, sweetcorn, sweet potato")
    st.code("paprika, jalepeño, ginger, garlic, peas, eggplant.")
    st.subheader("Content")
    st.text("This dataset contains three folders:")
    st.text("1. train (100 images each)")
    st.text("2. test (10 images each)")
    st.text("3. validation (10 images each)")

#prediction page
elif(app_mode=="Model Prediction"):
    st.header("Model predction")
    test_image = st.file_uploader("choose an image")
    if(st.button("Show Image")):
        st.image(test_image, width=4,use_column_width=True)
    # Predict Button
    if(st.button("Predict")):
        st.write("Our Prediction")
        st.snow()
        result_index = model_prediction(test_image)
        #Reading Lables
        with open("labels.txt")as f:
            content = f.readlines()
        label = []
        for i in content:
            label.append(i[:-1])
        st.success("Model is predicting it's a {}".format(label[result_index]))