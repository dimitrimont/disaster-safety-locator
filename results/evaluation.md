# Evaluation

170 landfalls from 137 storms (NOAA HRD, 1901-1970).

## The first version's split

The notebook's split (seed 42) scores R^2 0.79 (MAE 30.02 nm). Across 300 splits of the same kind the median is 0.347 (10th-90th percentile 0.16 to 0.54); 1 of 300 reach seed 42's score.

## Storms held out

| Model | MAE (nm) | median R^2 | always guessing the mean (nm) | splits |
|---|---|---|---|---|
| first forest, 13 inputs | 56.6 | 0.254 | 67.5 | 100 |
| revised forest, 6 storm measures | 54.2 | 0.294 | 67.5 | 100 |
| first forest, random rows (for comparison) | 51.5 | 0.34 | 66.5 | 100 |
| revised network, 6 storm measures | 61.3 | 0.062 | 67.5 | 20 |
| revised forest on the same splits | 54.6 | 0.271 | 67.5 | 20 |
