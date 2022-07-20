# Databricks notebook source
import pandas as pd
from prophet import Prophet

logging.getLogger('py4j').setLevel(logging.ERROR)

# COMMAND ----------

towerIds = spark.sql("select towerId from geospatial_tomasz.CDR_day_gold group by towerId")

# COMMAND ----------

def fitAndPlot(towerId_row):
  towerId = towerId_row["towerId"]
  
  model = Prophet(
    interval_width=.95,
    growth="linear",
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False
  )

  df = spark.sql("select CAST(datetime as date) as ds, sum(totalRecords_CDR) as y from geospatial_tomasz.CDR_day_gold where towerId = '{}' group by ds".format(towerId))
  pandas_df = df.toPandas()

  model.fit(pandas_df)

  future_pd = model.make_future_dataframe(
    periods=90, 
    freq='d', 
    include_history=True
    )

  forecast_pd = model.predict(future_pd)

  #trends_fig = model.plot_components(forecast_pd)
  #display(trends_fig)

  predict_fig = model.plot( forecast_pd, xlabel='date', ylabel="records {}".format(towerId))

  # adjust figure to display dates from last year + the 90 day forecast
  xlim = predict_fig.axes[0].get_xlim()
  new_xlim = ( xlim[1]-(180.0+10.0), xlim[1]-10.0)
  predict_fig.axes[0].set_xlim(new_xlim)

# COMMAND ----------

for index, row in towerIds.toPandas().iterrows():
  print(row)
  fitAndPlot(row)

# COMMAND ----------

# MAGIC %sql
# MAGIC select concat(dayofweek(datetime), "-", hour(datetime)) as day_hour, first(datetime) as datetime_, towerId, sum(totalRecords_CDR) from geospatial_tomasz.CDR_hour_gold where datetime >= "2021-07-11" and datetime < "2021-07-18" group by day_hour, towerId order by dayofweek(datetime_) asc, hour(datetime_) asc

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE geospatial_tomasz.CDR_one_week_ml AS select concat(dayofweek(datetime), "_", hour(datetime)) as day_hour, first(datetime) as dt, towerId, sum(totalRecords_CDR) sumCDR from geospatial_tomasz.CDR_hour_gold where datetime >= "2021-07-11" and datetime < "2021-07-18" group by day_hour, towerId order by dayofweek(dt) asc, hour(dt) asc

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE geospatial_tomasz.where datetime >= '2021-06-27' and datetime < '2021-07-18'

# COMMAND ----------

def fitAndPlotHourly(towerId_row):
  towerId = towerId_row["towerId"]
  
  model = Prophet(
    interval_width=.95,
    growth="linear",
    daily_seasonality=True,
    weekly_seasonality=True
  )

  df = spark.sql("select datetime as ds, totalRecords_CDR as y from geospatial_tomasz.CDR_hour_gold where datetime >= '2021-06-27' and datetime < '2021-07-18' and towerId = '{}'".format(towerId))
  pandas_df = df.toPandas()

  model.fit(pandas_df)

  future_pd = model.make_future_dataframe(
    periods=168, 
    freq='H', 
    include_history=True
    )

  forecast_pd = model.predict(future_pd)
  print(forecast_pd)

  #trends_fig = model.plot_components(forecast_pd)
  #display(trends_fig)

  predict_fig = model.plot( forecast_pd, xlabel='date', ylabel="records {}".format(towerId))

  return forecast_pd
  # adjust figure to display dates from last year + the 90 day forecast
  #xlim = predict_fig.axes[0].get_xlim()
  #new_xlim = ( xlim[1]-(7.0+10.0), xlim[1]-10.0)
  #predict_fig.axes[0].set_xlim(new_xlim)

# COMMAND ----------

forcastsByTower = []

for index, row in towerIds.toPandas().iterrows():
  print(row)
  forcastsByTower.append(fitAndPlotHourly(row))
  

# COMMAND ----------

print(forcastsByTower)

# COMMAND ----------

forcastsByTower[0]

# COMMAND ----------


