# NYC taxi demand forecasting

The locked replay established the update policy. The local production pipeline now records each zone-hour forecast with its model version and scores it after observed demand arrives. Repeated runs reuse verified predictions and scores. Experiment tracking records model lineage.
