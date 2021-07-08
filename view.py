import streamlit as st
from PIL import Image
from predict import Predicted
st.title("Upload + Classification Example")

predict = Predicted()
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=False)
    st.write("")
    st.write("Classifying...")
    label = predict.predict(image)
    st.write('%s' % (label))