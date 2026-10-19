# Wilmington hurricane impact: build a storm, predict how far its circulation reaches, and see
# which New Hanover County addresses are inside it.
#
#     streamlit run app.py
#
# Revised from the first version (tag `first-version`, then fake.py). What changed and why is in
# the README; the measurements are in results/evaluation.md.
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from hurricane.data import CATEGORY_WINDS
from hurricane.impact import OCI_RANGE_MB, WILMINGTON, exposure, rings, storm_row, summary
from hurricane.model import MODEL_DIR
from hurricane.model import load as load_forest

ROOT = Path(__file__).resolve().parent
# The county's address points ship zipped in the repo; the elevation model does not (see README).
GDB_PATH = os.environ.get("HURRICANE_GDB", f"zip://{ROOT}/data/GDB/NEWHANOVER-addresses-09-12-2024.zip!NEWHANOVER.gdb")
DEM_PATH = Path(os.environ.get("HURRICANE_DEM", ROOT / "data" / "newhanover-DEM03.tif"))

# Slider values per category (winds follow the Saffir-Simpson bands). The outer closed isobar is
# a pressure, about 1000-1020 mb for every storm in the data, so it no longer varies by category.
CAT_PRESETS = {
    1: {"Central_Pressure": range(980, 1001, 5), "RMW_nm": range(20, 60, 5)},
    2: {"Central_Pressure": range(965, 980, 5), "RMW_nm": range(15, 50, 5)},
    3: {"Central_Pressure": range(945, 965, 5), "RMW_nm": range(10, 40, 5)},
    4: {"Central_Pressure": range(920, 945, 5), "RMW_nm": range(8, 35, 5)},
    5: {"Central_Pressure": range(870, 921, 5), "RMW_nm": range(5, 30, 5)},
}


def middle(values) -> float:
    values = list(values)
    return float(values[len(values) // 2])


@st.cache_data(show_spinner="Loading address points...")
def load_addresses(gdb_path: str) -> pd.DataFrame:
    import geopandas as gpd

    try:
        gdf = gpd.read_file(gdb_path, layer="NEWHANOVER", columns=["Full_Address"])
    except Exception as e:  # the map still works without addresses
        st.warning(f"Address points could not be read from `{gdb_path}`: {e}")
        return pd.DataFrame(columns=["lon", "lat", "Full_Address", "elevation"])
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=2264)  # NC State Plane, feet
    gdf = gdf.to_crs(epsg=4326)
    return pd.DataFrame({"lon": gdf.geometry.x, "lat": gdf.geometry.y, "Full_Address": gdf["Full_Address"], "elevation": np.nan})


@st.cache_data(show_spinner="Sampling elevation...")
def attach_elevation(_df: pd.DataFrame, dem_path: str) -> pd.DataFrame:
    """Elevation from the DEM, or NaN where there is none (never a made-up 0 m)."""
    df = _df.copy()
    if df.empty or not Path(dem_path).exists():
        return df
    import rasterio
    from rasterio.warp import transform

    with rasterio.open(dem_path) as src:
        xs, ys = transform("EPSG:4326", src.crs, df["lon"].to_numpy(), df["lat"].to_numpy())
        values = np.array([v[0] for v in src.sample(zip(xs, ys))], dtype=float)
        if src.nodata is not None:
            values[values == src.nodata] = np.nan
    df["elevation"] = values
    return df


@st.cache_resource
def load_models() -> dict:
    models = {}
    if (MODEL_DIR / "size_forest.joblib").exists():
        models["Random forest"] = load_forest()
    try:
        from hurricane.deep import NETWORK_FILE, SizeNetwork

        if (MODEL_DIR / NETWORK_FILE).exists():
            models["Neural network"] = SizeNetwork.load(MODEL_DIR)
    except ImportError:  # TensorFlow is optional
        pass
    return models


def typical_miss() -> dict:
    path = ROOT / "results" / "evaluation.json"
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())["storms_held_out"]
    return {r["model"]: r["mae_nm"] for r in rows}


st.set_page_config(page_title="Wilmington Hurricane Impact", layout="wide")
st.title("Wilmington Hurricane Impact Simulator")

addresses = attach_elevation(load_addresses(GDB_PATH), str(DEM_PATH))
elevation_known = addresses["elevation"].notna().any()
models = load_models()

# ---- sidebar: the storm ----
st.sidebar.header("Hurricane Builder")
if not models:
    st.sidebar.error("No trained model in Hurricane_model/. Run `python scripts/train.py`.")
model_name = st.sidebar.selectbox("Model", list(models) or ["Random forest"])

sshws = st.sidebar.selectbox("Category (Saffir-Simpson)", [1, 2, 3, 4, 5], index=1)
lo, hi = CATEGORY_WINDS[sshws]
preset = CAT_PRESETS[sshws]
if st.session_state.get("category") != sshws:  # new category: start from its middle values
    st.session_state.update(
        category=sshws,
        max_winds=float((lo + hi) // 2),
        central_pressure=middle(preset["Central_Pressure"]),
        rmw=middle(preset["RMW_nm"]),
    )
max_winds = st.sidebar.slider("Max winds (kt)", float(lo), float(hi), step=1.0, key="max_winds")
central_pressure = st.sidebar.slider(
    "Central pressure (mb)", float(min(preset["Central_Pressure"])), float(max(preset["Central_Pressure"])), step=1.0, key="central_pressure"
)
rmw = st.sidebar.slider("Radius of max winds (nm)", float(min(preset["RMW_nm"])), float(max(preset["RMW_nm"])), step=1.0, key="rmw")
oci = st.sidebar.slider(
    "Outer closed isobar (mb)", float(OCI_RANGE_MB[0]), float(OCI_RANGE_MB[1]), 1010.0, step=1.0,
    help="The pressure at the outer edge of the circulation. Every storm in the data is 1001-1018 mb.",
)

st.sidebar.subheader("Landfall")
st.session_state.setdefault("lat", WILMINGTON[0])
st.session_state.setdefault("lon", WILMINGTON[1])
if "picked_center" in st.session_state:  # a click on the map, applied before the inputs are drawn
    st.session_state["lat"], st.session_state["lon"] = st.session_state.pop("picked_center")
if st.sidebar.button("Reset to Wilmington"):
    st.session_state["lat"], st.session_state["lon"] = WILMINGTON
lat = st.sidebar.number_input("Latitude", -90.0, 90.0, step=0.05, format="%.4f", key="lat")
lon = st.sidebar.number_input("Longitude", -180.0, 180.0, step=0.05, format="%.4f", key="lon")
st.sidebar.caption("Or click an address on the map to make it the landfall.")

st.sidebar.subheader("Layers")
show_addresses = st.sidebar.toggle("Show addresses", value=True)
show_rings = st.sidebar.toggle("Show rings", value=True)
color_by_risk = st.sidebar.toggle("Colour addresses by risk", value=False)
low_elev_ref = st.sidebar.slider("Low elevation below (m)", 0.0, 15.0, 5.0, step=0.5, disabled=not elevation_known)
if st.sidebar.button("Predict hurricane impact", type="primary"):
    st.session_state["predicting"] = True

# ---- prediction: live once asked for, so moving the landfall or a slider redraws it ----
size_nm = None
if st.session_state.get("predicting") and model_name in models:
    row = storm_row(lat, lon, max_winds, central_pressure, rmw, oci)
    size_nm = float(models[model_name].predict(row)[0])

center = (float(lat), float(lon))
data = addresses
counts = {"total": len(addresses), "inside": None, "low_elevation_inside": None}
if size_nm is not None and not addresses.empty:
    data = exposure(addresses, center, size_nm, low_elev_ref)
    counts = summary(data)

palette = {"High": [200, 0, 0], "Medium": [255, 140, 0], "Low": [0, 150, 0]}
if size_nm is not None and color_by_risk and not addresses.empty:
    data = data.assign(color=data["risk_level"].astype(str).map(palette).apply(lambda c: c if isinstance(c, list) else palette["Low"]))
else:
    data = data.assign(color=[[0, 0, 255]] * len(data))
data = data.assign(elevation_text=data["elevation"].round(1).astype(str).replace("nan", "unknown"))

layers = [pdk.Layer("TileLayer", data="https://services.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}", min_zoom=0, max_zoom=19, tile_size=256)]
if show_addresses and not data.empty:
    layers.append(pdk.Layer("ScatterplotLayer", id="addresses", data=data[["lon", "lat", "Full_Address", "elevation_text", "color"]],
                            get_position="[lon, lat]", get_radius=30, get_fill_color="color", pickable=True, auto_highlight=True))
if show_rings and size_nm is not None:
    colours = [[255, 215, 0, 25], [200, 0, 0, 60]]
    for (label, radius_m), colour in zip(rings(size_nm, rmw), colours):
        layers.append(pdk.Layer("ScatterplotLayer", data=pd.DataFrame({"lon": [lon], "lat": [lat], "r": [radius_m], "c": [colour]}),
                                get_position="[lon, lat]", get_radius="r", get_fill_color="c", stroked=True, filled=True,
                                line_width_min_pixels=2, pickable=False))

m1, m2, m3 = st.columns(3)
m1.metric("Addresses loaded", counts["total"])
m2.metric("Inside the circulation", "—" if counts["inside"] is None else counts["inside"])
m3.metric(f"Low (below {low_elev_ref:.1f} m) and inside",
          "needs a DEM" if not elevation_known else ("—" if counts["low_elevation_inside"] is None else counts["low_elevation_inside"]))

zoom = 10 if size_nm is None else float(np.clip(9.5 - np.log2(max(size_nm, 1) / 20), 4, 10))
deck = pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom, pitch=0), map_style=None,
                tooltip={"html": "<b>{Full_Address}</b><br/>Elevation (m): {elevation_text}", "style": {"backgroundColor": "white", "color": "black"}})
event = st.pydeck_chart(deck, height=600, on_select="rerun", selection_mode="single-object", key="impact_map")
picked = (event.selection.get("objects") or {}).get("addresses") if event and event.selection else None
if picked:
    new = (float(picked[0]["lat"]), float(picked[0]["lon"]))
    if new != st.session_state.get("last_pick"):  # a selection persists; act on it once
        st.session_state["last_pick"] = new
        st.session_state["picked_center"] = new
        st.rerun()

if size_nm is None:
    st.info("Build a storm in the sidebar, then press **Predict hurricane impact** to draw its rings.")
else:
    miss = typical_miss().get({"Random forest": "revised forest, 6 storm measures", "Neural network": "revised network, 6 storm measures"}[model_name])
    st.markdown(
        f"### Predicted size: **{size_nm:.0f} nm** to the outer closed isobar\n"
        f"Yellow: the predicted size (NOAA's *Size* is the average radius of the outer closed isobar). "
        f"Red: the radius of maximum winds you set, {rmw:.0f} nm."
        + (f"  \nOn storms it was not trained on, this model's size is off by {miss:.0f} nm on average." if miss else "")
    )
    if counts["inside"] == counts["total"] and counts["total"]:
        far = float(data["distance_nm"].max())
        st.caption(f"Every address is inside the circulation: the farthest is {far:.0f} nm from the landfall, and the smallest storm "
                   "in the data reached 75 nm. What separates addresses is how close they are to the landfall and how low they sit.")
if not elevation_known:
    st.caption(f"No elevation model at `{DEM_PATH}`: risk uses distance alone and no address is counted as low. See the README.")
