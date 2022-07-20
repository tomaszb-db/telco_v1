# Databricks notebook source
import pandas as pd
from prophet import Prophet
import logging

logging.getLogger('py4j').setLevel(logging.ERROR)

db_name = "geospatial_tomasz"
actualsHourlyGoldTable = "CDR_hour_gold"
forecastTable_Hourly = "telco_forecast_hourly"
anomolyTable_Hourly = "telco_anomly_hourly"

# COMMAND ----------

towerIds = spark.sql("select towerId from geospatial_tomasz.CDR_day_gold group by towerId")

# COMMAND ----------

#daily forecast for all towers
def fitAndPlotDaily(towerId_row):
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

#call the above function for each tower
for index, row in towerIds.toPandas().iterrows():
  print(row)
  fitAndPlotDaily(row)

# COMMAND ----------

#explore hourly for a week

# COMMAND ----------

# MAGIC %sql
# MAGIC select concat(dayofweek(datetime), "-", hour(datetime)) as day_hour, first(datetime) as datetime_, towerId, sum(totalRecords_CDR) from geospatial_tomasz.CDR_hour_gold where datetime >= "2021-07-11" and datetime < "2021-07-18" group by day_hour, towerId order by dayofweek(datetime_) asc, hour(datetime_) asc

# COMMAND ----------

#using hourly gold table to predict next two weeks activity
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

# COMMAND ----------

#calling above function
forcastsByTower = []

for index, row in towerIds.toPandas().iterrows():
  print(row)
  forcastsByTower.append(fitAndPlotHourly(row))

# COMMAND ----------

#do hourly prediction on all towers and save

#define schema output
from pyspark.sql.types import *
 
result_schema =StructType([
  StructField('ds',TimestampType()),
  StructField('towerId',StringType()),
  StructField('y',FloatType()),
  StructField('yhat',FloatType()),
  StructField('yhat_upper',FloatType()),
  StructField('yhat_lower',FloatType())
  ])

# COMMAND ----------

def forecastTowerHourly(history_pd):
  model = Prophet(
    interval_width=.95,
    growth="linear",
    daily_seasonality=True,
    weekly_seasonality=True
  )
  
  model.fit(history_pd)
  
  future_pd = model.make_future_dataframe(
    periods=168, 
    freq='H', 
    include_history=True
    )
  
  forecast_pd = model.predict(future_pd)
  
  f_pd = forecast_pd[ ['ds','yhat', 'yhat_upper', 'yhat_lower'] ].set_index('ds')
  
  h_pd = history_pd[['ds','towerId', 'y']].set_index('ds')
  
  results_pd = f_pd.join(h_pd, how='left')
  results_pd.reset_index(level=0, inplace=True)
  
  results_pd['towerId'] = history_pd['towerId'].iloc[0]
  
  return results_pd[['ds', 'towerId', 'y', 'yhat', 'yhat_upper', 'yhat_lower']]

# COMMAND ----------

#apply our forecast function to each tower

sql_statement = '''
  SELECT
    towerId,
    datetime as ds,
    sum(totalRecords_CDR) as y
  FROM geospatial_tomasz.CDR_hour_gold 
  WHERE datetime >= '2021-06-27' and datetime < '2021-07-18'
  GROUP BY towerId, ds
  ORDER BY towerId, ds
'''

tower_activity_hourly_history = (spark
                                   .sql(sql_statement)
                                   .repartition(sc.defaultParallelism, ["towerId"])
                                ).cache()

from pyspark.sql.functions import current_timestamp

results = (
  tower_activity_hourly_history
    .groupBy('towerId')
      .applyInPandas(forecastTowerHourly, schema=result_schema)
    .withColumn('training_date', current_timestamp() )
    )

#results.createOrReplaceTempView('new_forecasts')
results.write.format("delta").mode("append").saveAsTable("{}.{}".format(db_name, forecastTable_Hourly))

display(results)

# COMMAND ----------

#now for anomoly detection
#1) here we have a regular job every hour to calculate CDR metrics so that the last hour can be put through the anomoly detection function
currentTimeWindow = '2021-07-18T00:00:00.000+0000'
df_current = spark.sql("SELECT * FROM {}.{} WHERE datetime = '{}'".format(db_name, actualsHourlyGoldTable, currentTimeWindow))

def anomolyDetectUDF_Func(actual, forecast_min, forecast_max):
  if actual < forecast_min or actual > forecast_max:
    return 1
  else:
    return 0
  
def anomolyImportanceUDF_Func(actual, forecast_min, forecast_max):
  if actual < forecast_min:
    return forecast_min - actual
  elif actual > forecast_max:
    return actual - forecast_max
  else:
    return None
  
anomolyDetectUDF = F.udf(anomolyDetectUDF_Func)
anomolyImportanceUDF = F.udf(anomolyImportanceUDF_Func)

def isAnomoly(df_current):
  mostRecentTraining = spark.sql("SELECT training_date FROM geospatial_tomasz.telco_forecast_hourly GROUP BY training_date ORDER BY training_date DESC LIMIT 1")
  currentTrainingDatetime = mostRecentTraining.collect()[0]['training_date']
  currentTimeWindow = df_current.select(F.first("datetime")).collect()[0]["first(datetime)"]
  
  df_forecast = spark.sql("SELECT * FROM {}.{} WHERE training_date = '{}' AND ds = '{}'".format(db_name, forecastTable_Hourly, currentTrainingDatetime, currentTimeWindow))
  
  #anomoly logic
  df_current_plus_forecast = df_current.join(df_forecast, ["towerId"])
  df_current_plus_forecast_with_anomoly = df_current_plus_forecast
                                                     .withColumn("anomoly", anomolyDetectUDF(F.col("totalRecords_CDR"), F.col("yhat_lower"), F.col("yhat_upper")))
                                                     .withColumn("anomoly_importance", anomolyImportanceUDF(F.col("totalRecords_CDR"), F.col("yhat_lower"), F.col("yhat_upper")))
  
  return df_current_plus_forecast_with_anomoly
  
df_result = isAnomoly(df_current)
display(df_result)

# COMMAND ----------

import pyspark.sql.functions as F

currentTimeWindow = '2021-07-18T00:00:00.000+0000'
df_current = spark.sql("SELECT * FROM {}.{} WHERE datetime = '{}'".format(db_name, actualsHourlyGoldTable, currentTimeWindow))
currentTimeWindow_v2 = df_current.select(F.first("datetime")).collect()[0]["first(datetime)"]
mostRecentTraining = spark.sql("SELECT training_date FROM geospatial_tomasz.telco_forecast_hourly GROUP BY training_date ORDER BY training_date DESC LIMIT 1")
currentTrainingDatetime = mostRecentTraining.collect()[0]['training_date']
print(currentTimeWindow_v2)

df_forecast = spark.sql("SELECT * FROM {}.{} WHERE training_date = '{}' AND ds = '{}'".format(db_name, forecastTable_Hourly, currentTrainingDatetime, currentTimeWindow))

display(df_forecast)

# COMMAND ----------


mostRecentTraining = spark.sql("SELECT training_date FROM geospatial_tomasz.telco_forecast_hourly GROUP BY training_date ORDER BY training_date DESC LIMIT 1")
currentTraining = mostRecentTraining.collect()[0]['training_date']

df_forecast = spark.sql("SELECT * FROM {}.{} WHERE training_date = '{}'".format(db_name, forecastTable_Hourly, currentTraining))
display(df_forecast)

# COMMAND ----------

import mlflow.pyfunc

class ProphetModel(mlflow.pyfunc.PythonModel):
  import pandas as pd
  
  
