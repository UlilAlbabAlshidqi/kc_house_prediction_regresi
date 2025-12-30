import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pyclustering.utils.metric import distance_metric, type_metric

# Load komponen yang sudah disimpan
model = joblib.load('best_model_lgb.pkl')
preprocessor = joblib.load('preprocessor.pkl')
encoder = joblib.load('encoder.pkl')
columns_order = joblib.load('columns_order.pkl')
medoids = joblib.load('kmedoids_medoids.pkl')   # hanya medoids

# Daftar fitur input dasar
input_features = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'waterfront', 'view', 'condition', 'grade', 'sqft_above',
    'sqft_basement', 'sqft_living15', 'sqft_lot15',
    'age_since_renovation', 'is_renovated',
]

# Fungsi assign cluster berdasarkan medoid terdekat
def assign_cluster(lat, long, medoids):
    metric = distance_metric(type_metric.EUCLIDEAN)
    dists = [metric([lat, long], m) for m in medoids]
    return int(np.argmin(dists))

# Fitur turunan
def create_features(data):
    data['total_rooms'] = data['bedrooms'] + data['bathrooms']
    data['price_per_sqft'] = data['sqft_living'] / data['sqft_lot']
    data['rooms_per_floor'] = (data['bedrooms'] + data['bathrooms']) / data['floors']
    data['basement_ratio'] = data['sqft_basement'] / (data['sqft_living'] + 1e-6)
    data['livinglot_ratio'] = data['sqft_living'] / (data['sqft_lot'] + 1e-6)
    data['cluster'] = [assign_cluster(row['lat'], row['long'], medoids) for _, row in data.iterrows()]
    data = data.drop(columns=['lat','long'])
    return data

# Transform input agar sama dengan training
def transform_input(df_input):
    categorical_columns = ['waterfront', 'view', 'condition', 'grade', 'is_renovated', 'cluster']
    numerical_columns = [col for col in df_input.columns if col not in categorical_columns]
    
    # Log transform untuk numeric tertentu, kecuali 'age_since_renovation' dan 'is_renovated'
    cols_log = [
        'bedrooms', 'sqft_lot', 'sqft_above', 'sqft_basement',
        'sqft_living15', 'sqft_lot15', 'price_per_sqft',
        'rooms_per_floor', 'basement_ratio', 'livinglot_ratio'
    ]
    for col in cols_log:
        df_input[col] = df_input[col].apply(lambda x: np.log1p(max(x, 0)))
    
    # Scaling numeric
    X_num_scaled = pd.DataFrame(
        preprocessor.transform(df_input[numerical_columns]),
        columns=numerical_columns,
        index=df_input.index
    )
    
    # Encoding cluster
    cluster_encoded = encoder.transform(df_input[['cluster']])
    cluster_encoded_df = pd.DataFrame(
        cluster_encoded,
        columns=[f"cluster_{int(i)}" for i in range(cluster_encoded.shape[1])],
        index=df_input.index
    )
    
    # Gabungkan categorical selain cluster
    X_cat = pd.concat([df_input[categorical_columns].drop('cluster', axis=1), cluster_encoded_df], axis=1)
    
    # Final concat
    X_final = pd.concat([X_num_scaled.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    X_final = X_final.reindex(columns=columns_order, fill_value=0)
    return X_final

# ================= STREAMLIT APP =================
st.title("🏠 House Price Prediction")
st.write("Masukkan detail rumah untuk memprediksi harga:")

# Input form sesuai min-max + label jelas
bedrooms = st.number_input("Bedrooms (0–33)", min_value=1, max_value=33, value=3)
bathrooms = st.number_input("Bathrooms (0.0–8.0)", min_value=0.0, max_value=8.0, value=2.0)
sqft_living = st.number_input("Sqft Living (290–13540)", min_value=290, max_value=13540, value=1800)
sqft_lot = st.number_input("Sqft Lot (520–1651359)", min_value=520, max_value=1651359, value=5000)
floors = st.number_input("Floors (1.0–3.5)", min_value=1.0, max_value=3.5, value=1.0)
waterfront = st.selectbox("Waterfront", [0,1])
view = st.selectbox("View", [0,1,2,3,4])
condition = st.selectbox("Condition", [1,2,3,4,5])
grade = st.selectbox("Grade (1–13)", list(range(1,14)))
sqft_above = st.number_input("Sqft Above (290–9410)", min_value=290, max_value=9410, value=1500)
sqft_basement = st.number_input("Sqft Basement (0–4820)", min_value=0, max_value=4820, value=200)
sqft_living15 = st.number_input("Sqft Living 15 (399–6210)", min_value=399, max_value=6210, value=1800)
sqft_lot15 = st.number_input("Sqft Lot 15 (651–871200)", min_value=651, max_value=871200, value=5000)
age_since_renovation = st.number_input("Age Since Renovation (0–115)", min_value=0, max_value=115, value=10)

# Renovated? pakai Yes/No
is_renovated_str = st.selectbox("Renovated?", ["No", "Yes"])
is_renovated = 1 if is_renovated_str == "Yes" else 0

lat = st.number_input("Latitude (47.1559–47.7776)", min_value=47.1559, max_value=47.7776, value=47.5)
long = st.number_input("Longitude (-122.519–-121.315)", min_value=-122.519, max_value=-121.315, value=-122.0)

if st.button("Predict Price"):
    data_input = pd.DataFrame([[bedrooms, bathrooms, sqft_living, sqft_lot, floors,
                                waterfront, view, condition, grade, sqft_above,
                                sqft_basement, sqft_living15, sqft_lot15,
                                age_since_renovation, is_renovated,
                                lat, long]],
                              columns=input_features + ['lat','long'])
    
    data_input = create_features(data_input)
    X_ready = transform_input(data_input)
    price_pred = model.predict(X_ready)[0]
    st.success(f"💰 Predicted House Price: ${price_pred:,.2f}")
