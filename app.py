# =====================================================
# ENV & CONFIG
# =====================================================
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
st.set_page_config(
    page_title="India Air Quality Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.write("🚀 App started successfully")

# =====================================================
# IMPORTS
# =====================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

# =====================================================
# CONSTANTS
# =====================================================
POLLUTANTS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO",
    "SO2", "O3", "Benzene", "Toluene", "Xylene", "AQI"
]

CITY_COORDS = {
    "Thiruvananthapuram":[8.5241,76.9366], "Shillong":[25.5788,91.8933],
    "Jaipur":[26.9124,75.7873], "Mumbai":[19.0760,72.8777],
    "Ernakulam":[9.9816,76.2999], "Guwahati":[26.1445,91.7362],
    "Aizawl":[23.7271,92.7176], "Delhi":[28.7041,77.1025],
    "Bengaluru":[12.9716,77.5946], "Visakhapatnam":[17.6868,83.2185],
    "Lucknow":[26.8467,80.9462], "Patna":[25.5941,85.1376],
    "Kochi":[9.9312,76.2673], "Gurugram":[28.4595,77.0266],
    "Coimbatore":[11.0168,76.9558], "Amaravati":[16.5414,80.5150],
    "Chandigarh":[30.7333,76.7794], "Amritsar":[31.6340,74.8723],
    "Jorapokhar":[23.8,86.4], "Talcher":[20.9497,85.2332],
    "Kolkata":[22.5726,88.3639], "Hyderabad":[17.3850,78.4867],
    "Ahmedabad":[23.0225,72.5714], "Chennai":[13.0827,80.2707],
    "Bhopal":[23.2599,77.4126], "Brajrajnagar":[21.8160,83.9008]
}

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_dataset(folder="dataset"):
    if not os.path.exists(folder):
        return pd.DataFrame(), []

    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    dfs = []

    for f in files:
        try:
            df = pd.read_csv(os.path.join(folder, f))
            df["__source"] = f
            dfs.append(df)
        except:
            pass

    if not dfs:
        return pd.DataFrame(), files

    return pd.concat(dfs, ignore_index=True), files

# =====================================================
# PREPROCESS
# =====================================================
@st.cache_data
def preprocess(df):
    if df.empty:
        return df

    df = df.copy()
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["City", "Date"])

    present = []
    for c in POLLUTANTS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            present.append(c)

    df = df.sort_values(["City", "Date"])
    df[present] = df.groupby("City")[present].ffill().bfill()

    for c in present:
        df[c] = df[c].fillna(df[c].median())

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.weekday

    df["City"] = df["City"].astype("category")
    df["City_Code"] = df["City"].cat.codes

    return df.reset_index(drop=True)

# =====================================================
# HOME
# =====================================================
def page_home(df, files):
    st.title("🇮🇳 India Air Quality Explorer")
    st.success(f"Loaded {len(files)} file(s) | Total rows: {len(df):,}")

    st.markdown("""
    ### Purpose
    - Visualize air pollution trends  
    - Compare cities  
    - Explore pollution maps  
    - Predict AQI using ML  
    """)

# =====================================================
# DATA OVERVIEW
# =====================================================
def page_data_overview(df):
    st.header("📊 Data Overview")
    if df.empty:
        st.warning("No data available.")
        return
    st.dataframe(df.head(50))
    st.dataframe(df.describe().T)

# =====================================================
# EDA
# =====================================================
def page_eda(df):
    st.header("📈 Exploratory Data Analysis")
    if df.empty:
        st.warning("No data available.")
        return

    pollutant = st.selectbox("Pollutant", [p for p in POLLUTANTS if p in df.columns])
    fig, ax = plt.subplots()
    sns.histplot(df[pollutant], kde=True, ax=ax)
    st.pyplot(fig)

# =====================================================
# MAPS
# =====================================================
def page_maps(df):
    st.header("🗺️ AQI Maps")
    if df.empty:
        st.warning("No data available.")
        return

    df = df.copy()
    df["Latitude"] = df["City"].map(lambda c: CITY_COORDS.get(c, [None, None])[0])
    df["Longitude"] = df["City"].map(lambda c: CITY_COORDS.get(c, [None, None])[1])
    df = df.dropna(subset=["Latitude", "Longitude"])

    m = folium.Map(location=[22.97, 78.65], zoom_start=5)
    stats = df.groupby("City").agg({"AQI":"mean","Latitude":"first","Longitude":"first"}).reset_index()

    for _, r in stats.iterrows():
        folium.CircleMarker(
            [r["Latitude"], r["Longitude"]],
            radius=8,
            popup=f"{r['City']} AQI {r['AQI']:.1f}",
            color="red",
            fill=True
        ).add_to(m)

    st_folium(m, width=900, height=550)

# =====================================================
# MODEL (ALL FEATURES)
# =====================================================
def page_model(df):
    st.header("🤖 AQI Prediction Model")
    if df.empty:
        st.warning("No data available.")
        return

    FEATURES = [c for c in POLLUTANTS if c in df.columns and c != "AQI"]
    FEATURES += ["Year","Month","Day","Weekday","City_Code"]

    X = df[FEATURES].fillna(df[FEATURES].median())
    y = df["AQI"].fillna(df["AQI"].median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / np.clip(y_test, 1e-6, None))) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{rmse:.2f}")
    c2.metric("R²", f"{r2:.3f}")
    c3.metric("MAPE (%)", f"{mape:.2f}")

    st.subheader("🔮 Predict AQI")

    with st.form("predict"):
        city = st.selectbox("City", df["City"].unique())
        date = st.date_input("Date")

        inputs = {}
        for p in FEATURES:
            if p not in ["Year","Month","Day","Weekday","City_Code"]:
                inputs[p] = st.number_input(p, value=float(df[p].median()))

        submitted = st.form_submit_button("Predict")

    if submitted:
        date = pd.to_datetime(date)
        inputs["Year"] = date.year
        inputs["Month"] = date.month
        inputs["Day"] = date.day
        inputs["Weekday"] = date.weekday()

        city_map = dict(zip(df["City"].cat.categories, df["City"].cat.codes))
        inputs["City_Code"] = city_map[city]

        input_df = pd.DataFrame([inputs])[FEATURES]
        pred = model.predict(input_df)[0]

        st.success(f"🌟 Predicted AQI: {pred:.2f}")

# =====================================================
# MAIN ROUTER
# =====================================================
def main():
    df_raw, files = load_dataset()
    df = preprocess(df_raw)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Data Overview", "EDA", "Maps", "Model"]
    )

    if page == "Home":
        page_home(df, files)
    elif page == "Data Overview":
        page_data_overview(df)
    elif page == "EDA":
        page_eda(df)
    elif page == "Maps":
        page_maps(df)
    elif page == "Model":
        page_model(df)

# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    main()
