# === NEW: Vectorized RH and Heat Index functions + validation ===
import numpy as np
import logging

def compute_relative_humidity_vectorized(temp_c, dewpoint_c):
    """Vectorized RH from temp (°C) and dewpoint (°C). Returns percent [0-100]."""
    rh = 100.0 * (np.exp((17.625 * dewpoint_c)/(243.04 + dewpoint_c)) /
                  np.exp((17.625 * temp_c)/(243.04 + temp_c)))
    rh = np.clip(rh, 0, 100)
    return rh

def compute_heat_index_celsius(temp_c, rh):
    """
    Approximate Steadman/NOAA heat index (returns °C).
    Vectorized: temp_c, rh are numpy arrays or pandas Series.
    """
    # use the regression-style approximation in Celsius
    T = temp_c
    R = rh
    HI = (
        -8.784695 + 1.61139411*T + 2.338549*R
        - 0.14611605*T*R - 0.012308094*(T**2)
        - 0.016424828*(R**2) + 0.002211732*(T**2)*R
        + 0.00072546*T*(R**2) - 0.000003582*(T**2)*(R**2)
    )
    return HI

def validate_df_basic(df):
    """Quick sanity checks; raise/log if suspicious."""
    if df['temp'].isnull().any() or df['dewpoint'].isnull().any():
        logging.warning("Nulls found in temp or dewpoint")
    if (df['temp'] > 60).any() or (df['temp'] < -50).any():
        logging.warning("Temperature out of expected physical bounds")
    # more checks can be added

def add_weather_features(df):
    """Add computed weather features to the DataFrame."""
    # Assuming df has 't2m' for temp and 'd2m' for dewpoint (ERA5 variable names)
    # Convert from Kelvin to Celsius if needed
    if 't2m' in df.columns:
        df['temp'] = df['t2m'] - 273.15
    if 'd2m' in df.columns:
        df['dewpoint'] = df['d2m'] - 273.15

    # Compute RH and heat index
    df['rh'] = compute_relative_humidity_vectorized(df['temp'], df['dewpoint'])
    df['heat_index'] = compute_heat_index_celsius(df['temp'], df['rh'])

    # Validate
    validate_df_basic(df)

    return df
# === END NEW ===
