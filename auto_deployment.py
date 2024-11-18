import streamlit as st
import pickle
import pandas as pd

st.sidebar.title('Car Price Prediction')
html_temp = """
<div style="background-color:blue;padding:10px">
<h2 style="color:white;text-align:center;">Streamlit ML Cloud Auto Scout Project </h2>
</div>"""
st.markdown(html_temp,unsafe_allow_html=True)



model=st.sidebar.selectbox("Select model of your car", ('A1', 'A2', 'A3','Astra','Clio','Corsa','Espace','Insignia'))

year=st.sidebar.selectbox("Select year of car:",(2016,2017,2018,2019))

new_used=st.sidebar.selectbox("Select situation of your car", ("Used", "New", "Pre-registered","Employee's car","Demonstration"))

gearing_type=st.sidebar.radio('Select gear type',('Automatic','Manual','Semi-automatic'))

hp=st.sidebar.slider("Hp of car:", 51, 294, step=5)

gears=st.sidebar.slider("Gear of car", 5,9, step=1)

km=st.sidebar.slider("Km of car", 0,310000, step=1000)

weight=st.sidebar.slider("Weight of car", 900,3000, step=10)

displacement=st.sidebar.slider("Displacement of car:", 900, 3000, step=10)

conf_car = {"make_model": model,
           "hp": hp,
           "year": year,
           "gears": gears,
           "km": km,
           "weight": weight,
           "gearing_type": gearing_type, 
            "displacement": displacement, 
             "new_used": new_used} 

df = pd.DataFrame.from_dict([conf_car])

enc=pickle.load(open("ord_encoder","rb"))

df.loc[:,["make_model","gearing_type","new_used"]]

df.loc[:,["make_model","gearing_type","new_used"]] = enc.fit_transform(df.loc[:,["make_model","gearing_type","new_used"]])

st.header("The configuration of your car:")
st.table(df)

model=pickle.load(open("auto_scout","rb"))

# +
st.subheader("Press predict if configuration is okay")

if st.button("Predict"):
    prediction = model.predict(df)
    st.success("The estimated price of your car is €{} \u00B1 {}. ".format(int(prediction[0]), int(prediction[0])*0.0918))
