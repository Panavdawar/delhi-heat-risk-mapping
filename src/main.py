from _load_data import load_netcdf_to_df
from _preprocess import add_weather_features
from _risk_metrics import apply_risk_model
from spatial_mapping import spatial_join_weather
from visualization import create_risk_map



def main():
    df = load_netcdf_to_df("data/preprocessed/raw/delhi_weather_*.nc")
    df = add_weather_features(df)
    df = apply_risk_model(df)

    df = spatial_join_weather(
        df,
        "delhi_districts.geojson"
    )

    heat_map = create_risk_map(
        df,
        "delhi_districts.geojson"
    )

    heat_map.save("dwarka_heat_risk.html")

if __name__ == "__main__":
    main()
