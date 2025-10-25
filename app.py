import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, models, layers
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import altair as alt

# NEW SDK import
from google import genai
from google.genai import types as genai_types

# ------------------------ Load data and model ------------------------
df = pd.read_csv('cardekho_dataset.csv')

encoder = OneHotEncoder()
df_encoded = encoder.fit_transform(df[['brand','model', 'seller_type', 'fuel_type', 'transmission_type']])
odf = pd.DataFrame.sparse.from_spmatrix(df_encoded)
dff = df[['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'selling_price']]
cdf = pd.concat([odf, dff], axis=1)
cdf.columns = cdf.columns.astype(str)

scaler = StandardScaler()
scaler.fit(cdf)

scaler_y = StandardScaler()
scaler_y.fit(df[['selling_price']])

with open('xgboost_model_hyper1_lessoverfit.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# ------------------------ Styling ------------------------
st.markdown("""
<style>
@keyframes slide {
    0% { background: url('https://images.unsplash.com/photo-1488954048779-4d9263af2653?q=80&w=1470&auto=format&fit=crop&ixlib=rb-4.0.3') no-repeat center center fixed; }
    25% { background: url('https://images.unsplash.com/photo-1469050061383-f5fd48f3205d?q=80&w=1470&auto=format&fit=crop&ixlib=rb-4.0.3') no-repeat center center fixed; }
    50% { background: url('https://images.unsplash.com/photo-1465929517729-473000af12ce?q=80&w=1470&auto=format&fit=crop&ixlib=rb-4.0.3') no-repeat center center fixed; }
    75% { background: url('https://images.unsplash.com/photo-1488954048779-4d9263af2653?q=80&w=1470&auto=format&fit=crop&ixlib=rb-4.0.3') no-repeat center center fixed; }
    100% { background: url('https://images.unsplash.com/photo-1517026575980-3e1e2dedeab4?w=400&auto=format&fit=crop&q=60') no-repeat center center fixed; }
}
.main { position: relative; animation: slide 20s infinite; background-size: cover; background-position: center; }
.main::before { content: ''; position: absolute; top: 10; left: 10; width: 100%; height: 425%; background: rgba(0, 0, 0, 0.71); pointer-events: none; }
.content-box { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1); margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# ------------------------ App Title ------------------------
st.title("Know the correct price of your Car!")
st.write("Use this app to predict the selling price of your car based on various parameters.")
st.title("Enter the details to predict your car's price")

# ------------------------ User Input ------------------------
brand = st.selectbox("Enter brand", df['brand'].unique())
bbbb = df[df['brand'] == brand]
model = st.selectbox("Enter Model", bbbb['model'].unique())
aaaa = df[df['model'] == model]

aaaa['engine'] = [int(value) for value in aaaa['engine']]
aaaa['max_power'] = [int(value) for value in aaaa['max_power']]

vehicle_age = st.number_input("Enter age", value=3)
km_driven = st.number_input("Enter km driven", value=60000)
seller_type = 'Individual'
fuel_type = st.selectbox("Enter fuel type", df['fuel_type'].unique())
transmission_type = st.selectbox("Enter transmission type", df['transmission_type'].unique())
mileage = st.number_input("Whats is its mileage (Kmpl)?", value=21.5)
engine = st.selectbox("Enter Engine capacity (cc)", aaaa['engine'].unique())
max_power = st.selectbox("Max power in BHP", aaaa['max_power'].unique())
seats = st.selectbox("Number of seats", aaaa['seats'].unique())

if vehicle_age < 0:
    st.error("Vehicle age cannot be negative.")
if km_driven < 0:
    st.error("Kilometers driven cannot be negative.")

# ------------------------ Prepare Input ------------------------
X = [[brand, model , vehicle_age, km_driven, seller_type, fuel_type, transmission_type, mileage, engine, max_power, seats, 0]]
columns = ['brand','model', 'vehicle_age', 'km_driven', 'seller_type', 'fuel_type', 'transmission_type', 'mileage', 'engine', 'max_power', 'seats', 'selling_price']
X = pd.DataFrame(X, columns=columns)

X_enc = encoder.transform(X[['brand','model', 'seller_type', 'fuel_type', 'transmission_type']])
X_enc = pd.DataFrame.sparse.from_spmatrix(X_enc)

X_ndf = X[['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'selling_price']]
X_ccdf = pd.concat([X_enc, X_ndf], axis=1)
X_ccdf.columns = X_ccdf.columns.astype(str)

X_norm = scaler.transform(X_ccdf)
X_norm = pd.DataFrame(X_norm, columns=X_ccdf.columns)
X_norm = X_norm.drop(columns=['selling_price'])

X_for_tree = xgb.DMatrix(X_norm)
y = loaded_model.predict(X_norm)

y = pd.DataFrame(y)
y.columns = ['selling_price']
yd = scaler_y.inverse_transform(y)
yd = pd.DataFrame(yd)
yd.columns = ['selling_price']

st.header(f"Predicted Selling Price: ₹{int(yd['selling_price'][0])}")
st.write("Want to know how to find BHP of your car? Or have doubts regarding vehicle registration? Have a chat with our assistant bot on left.")

# ------------------------ Chatbot Section ------------------------
st.sidebar.title("Chat-bot")
GOOGLE_API_KEY = "AIzaSyBYRXjp8xyOiP9yzG-9Y0bgEJHIanCL9nY"

if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False
if "chat_session" not in st.session_state:
    st.session_state.chat_session = []

if st.sidebar.button("Send parameters to Gemini AI"):
    st.session_state.button_clicked = True

if st.session_state.button_clicked:
    client = genai.Client(api_key=GOOGLE_API_KEY)
    model_name = "gemini-2.0-flash"

    # Initial context
    initial_context = (
        f"You are a helpful car price assistant. The car brand is {brand}, "
        f"the model is {model}, the car's age is {vehicle_age}, "
        f"it has been driven {km_driven} km, and its predicted selling price is ₹{int(yd['selling_price'][0])}. "
        f"Please greet the user and summarize these inputs."
    )

    if not st.session_state.chat_session:
        response = client.models.generate_content(
            model=model_name,
            contents=genai_types.Content(
                role="user",
                parts=[genai_types.Part.from_text(text=initial_context)]
            )
        )
        st.session_state.chat_session.append(("Assistant", response.text))

    # Display chat history
    for role, text in st.session_state.chat_session:
        with st.sidebar:
            st.markdown(f"**{role}:** {text}")

    # User input
    user_prompt = st.sidebar.text_input("Ask Gemini...")
    if user_prompt:
        st.session_state.chat_session.append(("You", user_prompt))
        response = client.models.generate_content(
            model=model_name,
            contents=genai_types.Content(
                role="user",
                parts=[genai_types.Part.from_text(text=user_prompt)]
            )
        )
        st.session_state.chat_session.append(("Assistant", response.text))
        st.sidebar.markdown(f"**Assistant:** {response.text}")

# Clear chat
if st.sidebar.button("Clear Chat"):
    st.session_state.chat_session = []
    st.session_state.button_clicked = False
    st.sidebar.write("Chat cleared. Start a new conversation.")

# ------------------------ Find Cars Section ------------------------
st.subheader("")
st.subheader("Want to buy?")
st.subheader("Find cars under your requirements")

cost = st.number_input("Enter your max budget", value=500000)

col1, col2 = st.columns([3, 1])
with col1:
    vehicle_age_find = st.number_input("Enter maximum age", value=4)
with col2:
    vehicle_age_any_find = st.checkbox("Any", key="age_any")

col3, col4 = st.columns([3, 1])
with col3:
    km_driven_find = st.number_input("Enter max kilometers driven", value=50000)
with col4:
    km_driven_any_find = st.checkbox("Any", key="km_any")

if vehicle_age_any_find:
    vehicle_age_find = float('inf')
if km_driven_any_find:
    km_driven_find = float('inf')

ddde = df[(df['selling_price'] <= cost) & (df['vehicle_age'] <= vehicle_age_find) & (df['km_driven'] <= km_driven_find)]
ddde = ddde['car_name'].unique()

st.write("**Cars available under these filters:**")
if len(ddde) > 0:
    for car in ddde:
        st.write(f"• {car}")
else:
    st.write("No cars found matching your criteria.")

if st.button("Download List of cars"):
    ddde_df = pd.DataFrame(ddde, columns=['Car Name'])
    csv = ddde_df.to_csv(index=False)
    st.download_button(label="Get CSV", data=csv, file_name='car_list.csv', mime='text/csv')

st.markdown('</div>', unsafe_allow_html=True)
