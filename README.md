# Wilmington Hurricane Impact

Build a hurricane, predict how far its circulation will reach, and see which of New Hanover
County's 106,763 address points fall inside it and which of them sit low.

A CSC 432 (Applied Machine Learning) project at UNCW, Fall 2025, by Omari Harvey
([@OHarvey13](https://github.com/OHarvey13)), Joshua Bare ([@joshuabare](https://github.com/joshuabare))
and Dimitri Montgomery ([@dimitrimont](https://github.com/dimitrimont)). The team repository is
[OHarvey13/disaster-safety-locator](https://github.com/OHarvey13/disaster-safety-locator). This copy
is Dimitri's, with fixes made after the course; the code as it was submitted is the tag
`first-version`.

## What it does

`app.py` is a Streamlit app. In the sidebar you set a storm: its Saffir-Simpson category, maximum
winds, central pressure, radius of maximum winds and the pressure of its outer closed isobar, and
where it makes landfall (Wilmington by default; or click an address). A model trained on NOAA's
record of US landfalls predicts the storm's size, the radius of its outer closed isobar, and the map
draws two rings over the county's address points: that size, and the radius of maximum winds you
set. Tiles count the addresses inside the circulation and, if you add an elevation model, how many
of them sit below a height you choose. Addresses can be coloured by a simple risk score, closer to
the landfall and lower down being riskier.

Two models can be chosen: a random forest (the default) and a small neural network. On storms they
were not trained on, the forest's size is off by about 54 nm on average and the network's by about
61, against 68 for always guessing the average size. The app says so beside every prediction.

## Data

- **`Hurricane_dataset - Sheet1.csv`**: 170 landfalls, 1901 to 1970, from NOAA Hurricane Research
  Division's [Continental United States Hurricane Impacts/Landfalls](https://www.aoml.noaa.gov/hrd/hurdat/UShurrs_detailed.html).
  NOAA defines *Size* as "the average radius of the OCI" (the outer closed isobar), so it is a
  radius. A central pressure in parentheses there is an estimate; the spreadsheet stored those as
  negative numbers. A mark after a date is one of NOAA's footnotes.
- **`data/GDB/NEWHANOVER-addresses-09-12-2024.zip`**: New Hanover County's address points (a zipped
  FileGDB, read straight from the zip).
- **`data/raw/hurdat2_atlantic.txt`**: NOAA's HURDAT2 Atlantic track file, used by the notebooks
  in `notebooks/`.
- **Not included: an elevation model.** The first version sampled a county DEM,
  `newhanover-DEM03.tif`, that was never added to the repository. Put any GeoTIFF DEM at
  `data/newhanover-DEM03.tif`, or point `HURRICANE_DEM` at one. Without it the app says so, risk
  uses distance alone and no address is counted as low.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py              # the app
python scripts/train.py           # retrain both models into Hurricane_model/
python scripts/evaluate.py        # every number below, into results/ (about 10 minutes)
pytest                            # 14 tests, about 10 s
```

TensorFlow is only needed for the network; without it the app offers the forest alone
(`python scripts/train.py --no-network`).

## How good the size model is

Every evaluation here holds whole storms out: a storm that crosses several coasts has one row per
landfall, each with the same size, and a split that puts one landfall in training and another in
testing is testing the model on a storm it has seen. From `results/evaluation.md`:

| Model | Size off by, on average | Median R² | Always guessing the average |
|---|---|---|---|
| Random forest, revised (six storm measures) | 54.2 nm | 0.29 | 67.5 nm |
| Random forest, first version (13 inputs) | 56.6 nm | 0.25 | 67.5 nm |
| Neural network, revised (six storm measures) | 61.3 nm | 0.06 | 67.5 nm |

The revised forest is a little better than the first one on the same splits; the bigger change is
that its number is now an honest one. Its inputs are exactly what the app asks for. 170 landfalls are
not much: the model is better than guessing, not a forecast.

## What changed from the first version

**The model (`hurricane/`, `scripts/`).**

- **Its R² of 0.79 was one lucky split.** The notebook scored one random split (seed 42). Across
  300 splits made the same way the median scores 0.35 and no other reaches 0.79; with storms held
  out the forest scores about 0.25. Evaluation is now repeated over 100 splits that hold storms out.
- **19 of 170 dates were lost** to NOAA's footnote marks (`8/27/1909#`, `9/8/1934*`): the date
  parse turned them into blanks, which were filled with a median date. The loader keeps the date and
  the mark, and the model no longer uses dates at all: they let it recognise a storm from its other
  landfalls, and in the app they were always a made-up 15 September 1939.
- **An impossible pressure.** NOAA's table gives an outer isobar of 125 mb for 15 Sep 1945
  (the same storm's next row: 1011). It is now treated as missing.

**The app (`app.py`).**

- **Every ring was half size.** The app halved the predicted size as if it were a diameter. A storm
  predicted at 283.94 nm drew a 141.97 nm ring.
- **The outer-isobar slider did nothing.** It ran from 120 to 360 ("Outer Core Pressure Δ"), but the
  outer closed isobar is a pressure of about 1,000 to 1,020 mb, so every slider value was outside
  anything the model had seen and the prediction never moved, for any storm the app could build.
  It now takes millibars, and across the range in the data it moves the prediction by about 50 nm.
- **The rings could not tell addresses apart.** The inner rings were a third and two thirds of the
  halved size, which made the smallest ring the app could draw 31 nm, and every address is within
  16 nm of Wilmington. With the landfall there, all 106,763 addresses were inside every ring. The inner ring
  is now the radius of maximum winds you set, and the app says when the whole county is inside.
- **Clicking the map never moved the landfall.** `st.pydeck_chart` has no `on_click`, so the call fell
  back to a static map. Clicking an address now moves the landfall there.
- **Moving the landfall kept the old size.** The prediction depends on where the storm is; it is now
  recomputed whenever an input changes, once you have pressed Predict.
- **The low-elevation tile always showed "—"**, though the count was computed. It shows the count,
  or says a DEM is needed.
- **No elevation was treated as sea level.** Without a DEM every address got 0 m and counted as low.
- **A fresh clone could not find the model**, which the app looked for in
  `~/Downloads/CSC 432/Demo/`; paths are now inside the repository. A shape mismatch between the
  model and its inputs was hidden by padding with zeros; the input row is now built from the
  model's own feature list.
- `requirements.txt` listed PyTorch and AI Fairness 360, which nothing uses, and not Streamlit,
  geopandas, pydeck or rasterio, which the app needs.

**The neural network (`Deep_ML_Implementation/`, `hurricane/deep.py`).**

- **One scaler for two jobs.** The same MinMaxScaler was fitted to the inputs and then refitted to
  the sizes, so the inputs' scaling was lost and the saved network answered in 0-1 units (a raw
  input row gave -65). Inputs and sizes now have their own scalers, fitted on training rows only.
- **The row number was an input**, and rows with any missing value were dropped (54 of 170). The
  network now uses the forest's six storm measures and imputes missing values. Its app,
  `deep_ml_implemtatin.py`, passed 9 unscaled values to a network expecting 14; `app.py` now serves
  both models.
- Stopping its training early did not help (64 nm on the same splits).

**The track notebooks (`notebooks/`).**

- **Missing values were numbers.** HURDAT2 writes -999 for an unknown pressure, and 30,946 of the
  55,230 track points had one; the wind model was trained to predict -999 as a pressure. They are
  now missing.
- **Storm type gave the answer away.** `HU`, `TS` and `TD` are defined by wind speed, and they were
  inputs to the model predicting wind speed. They are no longer inputs, the split holds whole storms
  out, and the error is reported in mph and mb: winds off by 20.8 mph and pressure by 13.0 mb,
  against 23.5 and 14.5 for always guessing the average.

## Who did what

- **Omari Harvey**: the project's setup and proposal, the HURDAT2 processing notebook and the wind
  and pressure model (`notebooks/`).
- **Joshua Bare**: the GIS work, the address points, elevation sampling and the Wilmington flood
  layers (`Joshua_Bare_Code/`), the first Streamlit map, and the neural network
  (`Deep_ML_Implementation/`).
- **Dimitri Montgomery**: the landfall dataset, the random-forest size model
  (`Hurricane_Predicton_RandomForest.ipynb`), and the app's storm builder, rings, risk colouring and
  exposure tiles; and, in this copy, the fixes above, `hurricane/`, `scripts/` and the tests.

The proposal also planned shelters, FEMA's National Risk Index and a spatial deep-learning model of
safe zones. Those were not built.

## Tests

`tests/` checks the parts with a known answer: coordinates and NOAA's footnote dates parse, the
first row of the table and the 1945 pressure, storms are kept whole (Barbara's three landfalls are
one storm; the 4 and 5 September 1933 storms are two), no split puts one storm on both sides, one
degree of latitude is 60.04 nm, the rings use the full size, the model's input row is exactly its
features, an unknown elevation is not counted as low, the outer isobar moves the prediction, and the
network answers in nautical miles. The app was also run headless with Streamlit's `AppTest`, both
versions, with and without a DEM.

## Limits

Circles are a simplification: a real storm's wind field is lopsided, strongest to the right of its
track. The risk score is a heuristic, not a flood or wind model. The landfalls stop in 1970. Nothing
here should be used to decide whether to evacuate; follow the National Hurricane Center and local
emergency management.
