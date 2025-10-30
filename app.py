import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.title("💻 Laptop Price Prediction")
st.text("This app is built using a Machine Learning model")

# Load model and data
ml_model = pickle.load(open("pipe.pkl", "rb"))
df = pickle.load(open("df.pkl", "rb"))

# Inputs
co = st.selectbox("Select Company", df["Company"].unique(), index=3)
ty = st.selectbox("Select Type", df["TypeName"].unique(), index=3)
cpu = st.selectbox("Select CPU", df["Cpu"].unique(), index=3)
ram = st.radio("RAM (GB)", [8, 16, 32, 64], index=1)
gpu = st.selectbox("Select GPU", df["Gpu"].unique(), index=3)
os = st.selectbox("Select OS", df["OpSys"].unique(), index=3)
we = st.slider("Weight of Laptop (kg)", min_value=1.0, max_value=4.5, value=2.0, step=0.1)
ips = st.radio("IPS Display Available?", ["YES", "NO"], index=1)
touch = st.radio("Touchscreen Available?", ["YES", "NO"], index=1)
cpu_speed = st.slider("CPU Speed (GHz)", min_value=0.0, max_value=4.0, value=2.0, step=0.1)
hdd = st.radio("HDD (GB)", [256, 512, 1024, 2048], index=1)
ssd = st.radio("SSD (GB)", [256, 512, 1024, 2048], index=1)
ppi = st.slider("PPI", min_value=100.0, max_value=400.0, value=200.0, step=0.1)

# Prediction
if st.button("Predict Price"):
    ips = 1 if ips == "YES" else 0
    touch = 1 if touch == "YES" else 0

    query = pd.DataFrame([[co, ty, cpu, ram, gpu, os, we, ips, touch, cpu_speed, hdd, ssd, ppi]],
                         columns=["Company","TypeName","Cpu","Ram","Gpu","OpSys",
                                  "Weight","IPS","Touchscreen","Cpu_speed","HDD","SSD","ppi"])
    
    op = ml_model.predict(query)
    st.subheader(f"The Predicted Price of Laptop is ₹ {int(round(op[0], -2))}")
