import requests
import streamlit as st

# make a title
st.title('Text Classification App')

# make a text area
text = st.text_area('Enter some text')

# make a button
if st.button('Classify'):
    # make a request to the model by fastapi
    payload = {
        'text': text
    }
    
    print(payload)
    
    response = requests.post('http://172.18.0.2:8000/predict', json=payload)
    # get the prediction
    prediction = response.json()
    # show the prediction
    st.write(prediction)

# make a button to / endpoint
if st.button('Home'):
    response = requests.get('http://172.18.0.2:8000/')
    st.write(response.json())
    
    