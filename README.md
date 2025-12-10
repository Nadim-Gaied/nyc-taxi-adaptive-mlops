# NYC taxi demand forecasting

Models are compared on the same future windows with features available at forecast time. LightGBM with a Poisson objective outperformed the seasonal, ridge, and Poisson regression alternatives. Recent-demand features mattered; longer training histories did not show a reliable benefit over a rolling year. The model-update rules are fixed before the later historical replay.
