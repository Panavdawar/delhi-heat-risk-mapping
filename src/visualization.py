import folium

def create_risk_map(df, geojson_path):
    m = folium.Map(location=[28.59, 77.03], zoom_start=11)

    risk_avg = (
        df.groupby('district_name')['risk_label']
        .mean()
        .reset_index()
    )

    ### NEW ### defensive check
    if risk_avg.empty:
        raise ValueError("No risk data available for mapping")

    folium.Choropleth(
        geo_data=geojson_path,
        data=risk_avg,
        columns=['district_name', 'risk_label'],
        key_on='feature.properties.name',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Heat Risk Level'
    ).add_to(m)

    return m
