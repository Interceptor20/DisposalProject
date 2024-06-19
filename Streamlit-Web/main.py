import streamlit as st
import tensorflow as tf
import numpy as np
# from PIL import Image

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.h5')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64,64))  # Perform image preprocessing using the same target size as our test set
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image into batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Returns the index of the maximum probability class

# Define class names and disposal methods
class_names = ['Organic Waste', 'Recyclable-Inorganic Waste']
disposal_methods = {
    'Organic Waste': 'Compost or use as animal feed.',
    'Recyclable-Inorganic Waste': 'Recycle or dispose of in a landfill.'
}

# Sidebar
st.sidebar.title('Dashboard')
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'About Project', 'Waste Classification'])

# Main Page
if app_mode == 'Home':
    st.header('Project')
    st.subheader('Recognition and Classification of Organic and Inorganic Waste for Proper Recycling/Disposal Process')
    st.text('Built By: Michael Joshua')
    image_path = 'Streamlit-Web/home.jpg'
    st.image(image_path)

# About Project
elif app_mode == 'About Project':
    st.header('About Project')
    st.text('This Project Uses Machine Learning Models To Recognize and Classify Organic and Inorganic Waste For A Proper Recycling/Disposal Process')

    st.subheader('Problem')
    st.text('Waste management is a big problem in our country. Most of the wastes end up in landfills. This leads to many issues like:')
    st.code('Production of methane, Increase in landfills, Eutrophication, Consumption of toxic waste by animals, Leachate, Increase in toxins, Land, water and air pollution')

    st.subheader('About Dataset')
    st.text('Name: Waste Classification Data')
    st.text('This dataset contains 22500 images of organic and recyclable objects')
    st.text('Segregated into two classes (Organic and recyclable)')

    st.subheader('DataSet Link')
    st.text('https://www.kaggle.com/datasets/techsash/waste-classification-data')

    st.subheader('Content')
    st.text('Dataset is divided into train data (85%) and test data (15%)')
    st.text('Training data - 22564 images\nTest data - 2513 images')

# Waste Classification and Recognition Page
elif app_mode == 'Waste Classification':
    st.header('Waste Classification and Recognition')
    # test_image = st.file_uploader('Choose a Waste Material Image', type=['jpg', 'jpeg', 'png'])
    test_image = st.file_uploader('Choose Waste Material Image')
    if test_image is not None:
        if st.button('Show Waste Material'):
            st.image(test_image, use_column_width=True)
        # Prediction Button
        if st.button('Classify'):
            st.write("Model Waste Material Classification Result:")
            result_index = model_prediction(test_image)

            # Display the result
            print(result_index)
            result_class = class_names[result_index]
            disposal_method = disposal_methods[result_class]
            st.markdown(f"**Model is recognizing and classifying this material as {result_class}.**\n\n**Disposal method:** {disposal_method}")

