
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import page0,page1,page2
from PIL import Image

PAGES = {
    "Home" : page0,
    "Clustering" : page1,
    "Vizualization" : page2
}

image = Image.open('1_logo.png')
#st.image(image, caption='')
st.sidebar.title('INDIAN PREMIER LEAGUE')
st.sidebar.image(image,width=270)
selection = st.sidebar.radio("Select Here", list(PAGES.keys()))
page = PAGES[selection]
page.app()