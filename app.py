import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
st.set_page_config(
    page_title="India Air Quality Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")

st.write("🚀 App started successfully")

# -----------------------------
# CONSTANTS
# -----------------------------
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

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_dataset(folder="dataset"):
    if not os.path.exists(folder):
        return pd.DataFrame(), []

    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    dfs = []

    for f in files:
        try:
            df_tmp = pd.read_csv(os.path.join(folder, f))
            dfs.append(df_tmp)
        except:
            pass

    if not dfs:
        return pd.DataFrame(), files

    return pd.concat(dfs, ignore_index=True), files

# -----------------------------
# PREPROCESS
# -----------------------------
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
    df[present] = df.groupby("City")[present].transform(lambda g: g.ffill().bfill())
    df[present] = df[present].fillna(df[present].median())

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.weekday

    df["City"] = df["City"].astype("category")
    df["City_Code"] = df["City"].cat.codes

    return df.reset_index(drop=True)

# -----------------------------
# HOME PAGE
# -----------------------------
def page_home(df, files):
    st.title("🇮🇳 India Air Quality Explorer")

    st.success(f"Loaded {len(files)} file(s) | Total rows: {len(df):,}")

    st.markdown("""
### 🧪 Pollutants Tracked
- PM2.5, PM10  
- NO, NO2, NOx  
- CO, SO2, NH3  
- O3, Benzene, Toluene, Xylene  
- **AQI (Air Quality Index)**  

---
#### 🔍 *Use the sidebar to explore different sections*
""")

# -----------------------------
# DATA OVERVIEW
# -----------------------------
def page_data_overview(df):
    st.header("📊 Data Overview")

    if df.empty:
        st.warning("Dataset is empty!")
        return

    st.sidebar.subheader("Filters")

    cities = ["All"] + sorted(df["City"].unique())
    city_sel = st.sidebar.selectbox("Select City", cities)

    min_date, max_date = df["Date"].min(), df["Date"].max()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

    highlight_options = ["None"] + [c for c in POLLUTANTS if c in df.columns]
    highlight_index = highlight_options.index("AQI") if "AQI" in highlight_options else 0

    highlight_col = st.sidebar.selectbox(
        "Highlight Pollutant Column",
        highlight_options,
        index=highlight_index
    )

    df_f = df.copy()
    if city_sel != "All":
        df_f = df_f[df_f["City"] == city_sel]

    df_f = df_f[
        (df_f["Date"] >= pd.to_datetime(date_range[0])) &
        (df_f["Date"] <= pd.to_datetime(date_range[1]))
    ]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{len(df_f):,}")
    col2.metric("Cities", df_f["City"].nunique())
    col3.metric("Avg AQI", f"{df_f['AQI'].mean():.2f}")
    col4.metric("Date Span", f"{date_range[0]} → {date_range[1]}")

    st.subheader("📄 Data Preview")

    if highlight_col != "None":
        def highlight(x):
            return ['background-color:#ffe599' if x.name == highlight_col else '' for _ in x]
        st.dataframe(df_f.head(50).style.apply(highlight))
    else:
        st.dataframe(df_f.head(50))

# -----------------------------
# EDA (AQI DEFAULT)
# -----------------------------
def page_eda(df):
    st.header("📈 Exploratory Data Analysis")

    st.sidebar.subheader("EDA Controls")

    cities = ["All"] + sorted(df["City"].unique())
    city = st.sidebar.selectbox("City", cities)

    pollutants = [c for c in POLLUTANTS if c in df.columns]
    default_index = pollutants.index("AQI") if "AQI" in pollutants else 0

    pollutant = st.sidebar.selectbox(
        "Pollutant",
        pollutants,
        index=default_index
    )

    df_f = df if city == "All" else df[df["City"] == city]

    yearly = df_f.groupby("Year")[pollutant].mean()

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(yearly.index, yearly.values, marker="o")
    ax.set_title(f"Yearly Trend of {pollutant}")
    ax.grid(True)
    st.pyplot(fig)

# -----------------------------
# MAPS (AQI DEFAULT)
# -----------------------------
def page_maps(df):
    st.header("🗺️ AQI Maps")

    pollutant_list = [c for c in POLLUTANTS if c in df.columns]
    default_index = pollutant_list.index("AQI") if "AQI" in pollutant_list else 0

    pollutant_choice = st.sidebar.selectbox(
        "Pollutant",
        pollutant_list,
        index=default_index
    )

    df["Lat"] = df["City"].map(lambda c: CITY_COORDS.get(str(c), [None,None])[0])
    df["Lon"] = df["City"].map(lambda c: CITY_COORDS.get(str(c), [None,None])[1])
    df = df.dropna(subset=["Lat","Lon"])

    m = folium.Map(location=[22.97, 78.65], zoom_start=5)

    stats = df.groupby("City").mean(numeric_only=True)

    for city, r in stats.iterrows():
        folium.CircleMarker(
            location=[r["Lat"], r["Lon"]],
            radius=10,
            popup=f"{city}<br>{pollutant_choice}: {r[pollutant_choice]:.1f}",
            fill=True,
            fill_opacity=0.7
        ).add_to(m)

    st_folium(m, width=900, height=500)

# -----------------------------
# MODEL
# -----------------------------
def page_model(df):
    st.header("🤖 AQI Prediction Model")

    FEATURES = [c for c in POLLUTANTS if c != "AQI" and c in df.columns]
    FEATURES += ["Year", "Month", "Day", "Weekday", "City_Code"]

    X = df[FEATURES]
    y = df["AQI"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    st.metric("RMSE", f"{rmse:.2f}")
    st.metric("R² Score", f"{r2:.3f}")

# -----------------------------
# MAIN
# -----------------------------
df, files = load_dataset()
df = preprocess(df)

page = st.sidebar.radio(
    "Navigation",
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

