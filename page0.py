import streamlit as st
from PIL import Image

def app():
    
    modelTraining = st.container()

    with modelTraining:
        new_title = '<p style="font-family:sans-serif; color:white; font-size: 34px;margin-left:200px;">IPL (2008-2020)</p>'
        st.markdown(new_title,unsafe_allow_html=True)

        image = Image.open('home_page.jpg')
        st.image(image, caption='indian premier league')
        
        new_text = '<div> <div>The Indian Premier League (IPL) is a professional Twenty20 cricket league, contested by eight teams based out of eight different Indian cities.[3] The league was founded by the Board of Control for Cricket in India (BCCI) in 2007. It is usually held between March and May of every year and has an exclusive window in the ICC Future Tours Programme.</div> <br> <div>In this project, we have worked on IPL Data Analysis and Visualization using Python where we will explore interesting insights from the data of IPL matches with data from IPL seasons 2008–2020.</div> </div>'
        st.markdown(new_text,unsafe_allow_html=True)