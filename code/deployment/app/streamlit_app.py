import requests
import streamlit as st

# make a title
st.title('Text Classification App')

# make a text area
text = st.text_area('Enter some text')

# make a button
if st.button('Classify'):
    # make a request to the model by fastapi
    response = requests.post('http://localhost:8000/predict', json={'text': text})
    # get the prediction
    prediction = response.json()
    # show the prediction
    st.write(prediction)
    
    