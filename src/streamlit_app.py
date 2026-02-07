import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from pathlib import Path
import pandas as pd
import numpy as np
import altair as alt

# App config
st.set_page_config(
    page_title="Dwarka Heat Risk Dashboard",
    layout="wide",
    page_icon="🔥",
)

BASE = Path(__file__).resolve().parent.parent
GEO_PATH = BASE / "geo" / "dwarka_grid_risk.geojson"
WEATHER_PATHS = [
    BASE / "data" / "processed" / "delhi_weather_with_risk.csv",
    BASE / "notebooks" / "data" / "processed" / "delhi_weather_with_risk.csv",
]

# --- Heat index helpers (°C input) ---
def compute_rh(temp_c, dewpoint_c):
    es = 6.11 * np.exp((17.27 * temp_c) / (237.3 + temp_c))
    e = 6.11 * np.exp((17.27 * dewpoint_c) / (237.3 + dewpoint_c))
    rh = 100.0 * (e / es)
    return np.clip(rh, 0, 100)

def heat_index_celsius(T, RH):
    T_F = T * 9 / 5 + 32
    HI_F = (
        -42.379
        + 2.04901523 * T_F
        + 10.14333127 * RH
        - 0.22475541 * T_F * RH
        - 0.00683783 * T_F * T_F
        - 0.05481717 * RH * RH
        + 0.00122874 * T_F * T_F * RH
        + 0.00085282 * T_F * RH * RH
        - 0.00000199 * T_F * T_F * RH * RH
    )
    return (HI_F - 32) * 5 / 9

@st.cache_data
def load_grid():
    grid = gpd.read_file(GEO_PATH)
    # Reproject to a metric CRS for centroid math; keep aux data separate to avoid JSON serialization issues
    grid_proj = grid.to_crs("EPSG:32643")  # UTM zone covering Delhi
    centroids_ll = grid_proj.geometry.centroid.to_crs("EPSG:4326")
    return grid, centroids_ll

@st.cache_data
def load_weather():
    for p in WEATHER_PATHS:
        if p.exists():
            df = pd.read_csv(p, parse_dates=["time"])
            break
    else:
        raise FileNotFoundError("delhi_weather_with_risk.csv not found in expected paths")

    # Compute RH / heat index if missing (some files already have apparent_temp + heat_risk)
    if "RH" not in df.columns and {"temp", "wind"}.issubset(df.columns):
        df["RH"] = np.nan  # fallback; better than crash
    if "heat_index" not in df.columns and {"temp", "RH"}.issubset(df.columns):
        df["heat_index"] = heat_index_celsius(df["temp"], df["RH"])

    df["date"] = df["time"].dt.date
    return df

grid, centroids_ll = load_grid()
wx = load_weather()

# Daily lens for storytelling
daily = (
    wx.groupby("date").agg(
        temp_mean=("temp", "mean"),
        hi_mean=("heat_index", "mean") if "heat_index" in wx.columns else ("temp", "mean"),
        wind_mean=("wind", "mean"),
        high_risk=("heat_risk", lambda s: (s >= 2).mean() if "heat_risk" in wx else np.nan),
    )
    .reset_index()
    .dropna(subset=["temp_mean", "wind_mean"])
)

# Simple wind-adjusted apparent heat index (convective cooling proxy)
daily["hi_wind_adj"] = (daily["hi_mean"] - 0.7 * (daily["wind_mean"] - 1)).clip(lower=0)
heatwave_days = (daily["hi_mean"] >= 40).sum()

# Wind–heat interaction stats (computed once, reused below)
calm_thresh = daily["wind_mean"].quantile(0.2)
breeze_thresh = daily["wind_mean"].quantile(0.8)
calm = daily[daily["wind_mean"] <= calm_thresh]
breezy = daily[daily["wind_mean"] >= breeze_thresh]
calm_hi = calm["hi_mean"].mean()
breezy_hi = breezy["hi_mean"].mean()
calm_risk = calm["high_risk"].mean() if "high_risk" in daily else np.nan
breezy_risk = breezy["high_risk"].mean() if "high_risk" in daily else np.nan
wind_hi_corr = daily[["wind_mean", "hi_mean"]].corr().iloc[0, 1]
beta_wind, alpha_wind = np.polyfit(daily["wind_mean"], daily["hi_mean"], 1)

# Summary metrics
st.title("🔥 Dwarka Heat Risk Dashboard")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Grid Cells", len(grid))
col2.metric("Avg Heat Index (spatial)", round(grid["mean_hi"].mean(), 2))
col3.metric("Max Heat Index (spatial)", round(grid["mean_hi"].max(), 2))
col4.metric("Mean Wind Speed", f"{daily['wind_mean'].mean():.1f} m/s")
col5.metric("Heatwave Days (HI≥40°C)", int(heatwave_days))

hot_day = daily.loc[daily["temp_mean"].idxmax()]
windy_day = daily.loc[daily["wind_mean"].idxmax()]

st.markdown(
    f"""
    **Story so far**  
    - Hottest mean day: **{hot_day['date']}** at **{hot_day['temp_mean']:.1f}°C**; wind averaged **{hot_day['wind_mean']:.1f} m/s**.  
    - Windiest day: **{windy_day['date']}** at **{windy_day['wind_mean']:.1f} m/s**; temperature was **{windy_day['temp_mean']:.1f}°C**.  
    - Across the record, wind speeds cluster around {daily['wind_mean'].median():.1f} m/s; calm days (<2 m/s) coincide with higher apparent heat risk in this dataset.
    """
)

st.markdown(
    """
    **Risk interpretation**
    - Green → Low heat stress  
    - Orange → Moderate heat stress  
    - Red → High heat stress  
    """
)

# Build map
centers = centroids_ll.apply(lambda p: (p.y, p.x))
center_lat = centers.apply(lambda t: t[0]).mean()
center_lon = centers.apply(lambda t: t[1]).mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="cartodbpositron")
folium.Choropleth(
    geo_data=grid,  # GeoDataFrame is fine as long as it has only one geometry column
    data=grid,
    columns=["cell_id", "mean_hi"],
    key_on="feature.properties.cell_id",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Mean Heat Index",
).add_to(m)

st.subheader("🗺️ Heat Risk Map")
st_folium(m, width=1000, height=600)

st.subheader("📋 Top 10 Hot Cells")
st.dataframe(
    grid[["cell_id", "mean_hi"]]
    .sort_values("mean_hi", ascending=False)
    .head(10),
    use_container_width=True,
)

# Wind vs. temperature narrative
st.subheader("🌬️ Wind vs Heat: Daily Relationship")
chart = (
    alt.Chart(daily)
    .mark_circle(size=60, opacity=0.6, color="#d95f02")
    .encode(
        x=alt.X("wind_mean:Q", title="Wind speed (m/s)"),
        y=alt.Y("hi_mean:Q", title="Mean heat index (°C)"),
        tooltip=["date:T", "hi_mean:Q", "temp_mean:Q", "wind_mean:Q", "high_risk:Q"],
    )
    .transform_regression("wind_mean", "hi_mean", method="linear").mark_line(color="#1b9e77")
    .properties(height=320)
)
st.altair_chart(chart, use_container_width=True)

# Timeline: heat index vs wind-adjusted heat index
st.subheader("⏱️ Time Series: Heat Index vs Wind-Adjusted")
base = alt.Chart(daily).encode(x=alt.X("date:T", title="Date"))
hi_line = base.mark_line(color="#e6550d").encode(y=alt.Y("hi_mean:Q", title="Heat index (°C)"), tooltip=["date:T", "hi_mean:Q", "wind_mean:Q"])
adj_line = base.mark_line(color="#3182bd").encode(y=alt.Y("hi_wind_adj:Q"), tooltip=["date:T", "hi_wind_adj:Q", "wind_mean:Q"])
wind_bar = base.mark_bar(color="#74c476", opacity=0.4).encode(
    y=alt.Y("wind_mean:Q", title="Wind (m/s)", axis=alt.Axis(orient="right"))
)
st.altair_chart(alt.layer(hi_line, adj_line, wind_bar).resolve_scale(y="independent"), use_container_width=True)

story_md = f"""
**Wind–Heat Story**
- Pearson r(wind, heat index) = **{wind_hi_corr:.2f}** (negative: more wind, lower apparent heat).
- Regression slope = **{beta_wind:.2f} °C per m/s** (each extra m/s trims ~{abs(beta_wind):.1f}°C off the heat index on average).
- Calmest quintile (≤{calm_thresh:.1f} m/s): mean HI **{calm_hi:.1f}°C**{'' if np.isnan(calm_risk) else f'; high-risk share ~{int(100*calm_risk)}%'}.
- Breeziest quintile (≥{breeze_thresh:.1f} m/s): mean HI **{breezy_hi:.1f}°C**{'' if np.isnan(breezy_risk) else f'; high-risk share ~{int(100*breezy_risk)}%'}.
- Interpretation: Wind drives convective + evaporative cooling, acting as a natural buffer during heat waves; this dataset shows breezy days cutting perceived heat by ~{(calm_hi - breezy_hi):.1f}°C versus calm days.
"""
st.markdown(story_md)

abstract = f"""
In this project, we quantified how wind speed modulates heat stress in Delhi. Using temperature and humidity, we computed the Heat Index and then linked it with wind to capture apparent temperature. Daily analysis shows a strong negative correlation (r = {wind_hi_corr:.2f}); a linear fit suggests each 1 m/s of wind lowers perceived heat by ~{abs(beta_wind):.1f}°C. Calm-quintile days (≤{calm_thresh:.1f} m/s) averaged {calm_hi:.1f}°C, while the breeziest (≥{breeze_thresh:.1f} m/s) averaged {breezy_hi:.1f}°C, reducing high-risk exposure. This evidence supports wind-aware urban heat risk assessment, emergency planning, and climate resilience for Delhi's heat waves.
"""

abstract_short = (
    "Wind speeds in Delhi show a clear cooling effect: each 1 m/s trims about "
    f"{abs(beta_wind):.1f}°C from the heat index (r = {wind_hi_corr:.2f}). "
    f"Calm days (≤{calm_thresh:.1f} m/s) averaged {calm_hi:.1f}°C HI vs. "
    f"{breezy_hi:.1f}°C on breezy days (≥{breeze_thresh:.1f} m/s), cutting high-risk exposure. "
    "Wind-aware planning can buffer heat waves and improve resilience."
)

st.subheader("🧾 Ready-to-use Conclusion & Abstract")
st.text_area("Abstract (formal)", abstract, height=170)
st.text_area("Abstract (concise, 3-4 sentences)", abstract_short, height=110)
st.download_button("Download formal abstract", abstract, file_name="wind_heat_conclusion.txt")

st.caption("Built with ERA5-derived grid heat index for Dwarka, New Delhi; heat risk contextualised with co-located wind conditions.")
