import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score



st.title("""DIAMOND PRICE PREDICTION APP

This app predicts the price of diamond based on physical attributes """
)



st.sidebar.subheader('User Input Parameters')

def user_input_features():
    carat=st.sidebar.slider("Carat",0.0,5.0,0.31)
    cut=st.sidebar.select_slider("Cut",options=['Ideal', 'Premium', 'Good', 'Very Good', 'Fair'])
    color=st.sidebar.select_slider("Color",options=['E', 'I', 'J', 'H', 'F', 'G', 'D'])
    clarity=st.sidebar.select_slider("Clarity",options=['SI2', 'SI1', 'VS2', 'VVS2', 'VVS1', 'VS1', 'I1', 'IF'])
    depth=st.sidebar.slider("Depth",10.0,100.0,60.0)
    table=st.sidebar.slider("Table",10.0,100.0,50.0)
    x=st.sidebar.slider("X-Dimension",0.1,10.0,0.5)
    y=st.sidebar.slider("y-Dimension",0.1,10.0,0.5)
    z=st.sidebar.slider("Z-Dimension",0.1,10.0,0.5)
    data={"carat":carat,"cut":cut,"color":color,"clarity":clarity,"depth":depth,"table":table,"x":x,"y":y,"z":z}
    features=pd.DataFrame(data,index=[0])
    return features


df=user_input_features()


cut_map={'Fair':1,'Good':2,'Very Good':3,'Premium':4,'Ideal':5}
df['cut']=df['cut'].map(cut_map)

color_map={'D':7,'E':6,'F':5,'G':4,'H':3,'I':2,'J':1}
df['color']=df['color'].map(color_map)

clarity_map={'I1':1,'SI2':2, 'SI1':3,'VS2':4,'VS1':5,'VVS2':6, 'VVS1':7,'IF':8}
df['clarity']=df['clarity'].map(clarity_map)

scalar=pickle.load(open("scaling.pkl","rb"))
reg=pickle.load(open("reg.pkl","rb"))

test_data=scalar.transform(df)

predict=reg.predict(test_data)



st.markdown("---")
st.write("""

## PREDICTED PRICE $ """+str(predict[0]))

st.markdown("---")

st.subheader("""Many times you heard about 4C’s of diamond as these work as a determinant of a diamond price. Now I will shed light on each C just to make sure that you have gotten the adequate details of each feature.""")
  

st.write("""
  **1st Characteristic of Diamond: Carat**
      Carat is weight of diamond. One carat is 1/5 of a gram. The weight or carat of diamond is an important factor that defines its cost. One carat is divided into 100 equal points. Generally, people love to buy one carat (100 points) diamond as it is regarded the best. If you buy a diamond with a weight of 0.99 carats then you will find a decrease in diamond value as point different matters a lot.

  **2nd Characteristic of Diamond: Color**

   The colour of gem-quality diamonds occurs in many hues. In the range from colourless to light yellow or light brown. Colourless diamonds are the rarest. Other natural colours (blue, red, pink for example) are known as "fancy,” and their colour grading is different than from white colorless diamonds.  
   from J (worst) to D (best)

   **3rd Characteristic of Diamond: Clarity**

   Diamonds can have internal characteristics known as inclusions or external characteristics known as blemishes. Diamonds without inclusions or blemishes are rare; however, most characteristics can only be seen with magnification.  
   I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best)

   **4th Characteristic of Diamond: Cut**

   No doubt, Cut is the most important characteristic of the diamond. It determines how the light which enters into the diamond from the above will be reflected back to the eye of observer. A perfect cut diamond reflects light to its optimum. However, when you get a diamond with deep cut, it will not show the extreme brightness; it seems that light leaks from it.

""")


st.write(':gem: Hope you gained Knowledge and had fun') 






