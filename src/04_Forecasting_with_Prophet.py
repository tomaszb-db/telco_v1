# Databricks notebook source
import pandas as pd
from prophet import Prophet

# COMMAND ----------

model = Prophet(
  interval_width=.95,
  growth="linear",
  daily_seasonality=True,
  weekly_seasonality=True,
  yearly_seasonality=False
)

df = spark.sql("select CAST(datetime as date) as ds, sum(totalRecords_CDR) as y from geospatial_tomasz.CDR_day_gold group by ds")
pandas_df = df.toPandas()

model.fit(pandas_df)

# COMMAND ----------

future_pd = model.make_future_dataframe(
  periods=90, 
  freq='d', 
  include_history=True
  )
 

# COMMAND ----------

forecast_pd = model.predict(future_pd)

# COMMAND ----------

trends_fig = model.plot_components(forecast_pd)
display(trends_fig)

# COMMAND ----------

display(forecast_pd)

# COMMAND ----------

predict_fig = model.plot( forecast_pd, xlabel='date', ylabel='records')
 
# adjust figure to display dates from last year + the 90 day forecast
xlim = predict_fig.axes[0].get_xlim()
new_xlim = ( xlim[1]-(90.0+10.0), xlim[1]-10.0)
predict_fig.axes[0].set_xlim(new_xlim)
 
#display(predict_fig)

# COMMAND ----------

pd.DataFrame

# COMMAND ----------

import pandas as pd

def forecast_tower_activity(history_pd):
  model = Prophet(
    interval_width= 0.95,
    growth='linear',
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False
  )
  
  model.fit(history_pd)
  
  future_pd = model.make_future_dataframe(
    periods=90,
    freq="d",
    include_history=True
  )
  
  results_pd = model.predict(future_pd)
  
  

  f_pd = forecast_pd[ ['ds','yhat', 'yhat_upper', 'yhat_lower'] ].set_index('ds')

  h_pd = history_pd[["ds", "towerId", "y"]].set_index("ds")
  
  results_pd = f_pd.join(h_pd, how="left")
  results_pd.reset_index(level=0, inplace=True)
  
  results_pd["towerId"] = history_pd["towerId"].iloc[0]

  
  return results_pd[ ['ds', 'towerId', 'y', 'yhat', 'yhat_upper', 'yhat_lower'] ]  

# COMMAND ----------

df_history = spark.sql("select CAST(datetime as date) as ds, sum(totalRecords_CDR) as y, towerId from geospatial_tomasz.CDR_day_gold group by towerId, datetime")
display(df_history)

# COMMAND ----------

from pyspark.sql.types import *

result_schema =StructType([
  StructField('ds',DateType()),
  StructField('towerId', StringType()),
  StructField('y',FloatType()),
  StructField('yhat',FloatType()),
  StructField('yhat_upper',FloatType()),
  StructField('yhat_lower',FloatType())
  ])

# COMMAND ----------

from pyspark.sql.functions import current_date


results = (
  df_history
    .groupBy("towerId")
      .applyInPandas(forecast_tower_activity, schema=result_schema)
    .withColumn("training_date", current_date())
)

results.createOrReplaceTempView('new_forecasts')

display(results.filter("ds > \'2021-12-31\' and ds < \'2022-12-01\'"))



# COMMAND ----------

# schema of expected result set
eval_schema =StructType([
  StructField('training_date', DateType()),
  StructField('towerId', StringType()),
  StructField('mae', FloatType()),
  StructField('mse', FloatType()),
  StructField('rmse', FloatType())
  ])

# define function to calculate metrics
def evaluate_forecast( evaluation_pd: pd.DataFrame ) -> pd.DataFrame:
  
  # get store & item in incoming data set
  training_date = evaluation_pd['training_date'].iloc[0]
  towerId = evaluation_pd['towerId'].iloc[0]
  
  # calulate evaluation metrics
  mae = mean_absolute_error( evaluation_pd['y'], evaluation_pd['yhat'] )
  mse = mean_squared_error( evaluation_pd['y'], evaluation_pd['yhat'] )
  rmse = sqrt( mse )
  
  # assemble result set
  results = {'training_date':[training_date], 'towerId':[towerId], 'mae':[mae], 'mse':[mse], 'rmse':[rmse]}
  return pd.DataFrame.from_dict( results )

# calculate metrics
results = (
  spark
    .table('new_forecasts')
    .filter('ds < \'2018-01-01\'') # limit evaluation to periods where we have historical data
    .select('training_date', 'towerId', 'y', 'yhat')
    .groupBy('training_date', 'towerId')
    .applyInPandas(evaluate_forecast, schema=eval_schema)
    )

results.createOrReplaceTempView('new_forecast_evals')

# COMMAND ----------


test_pd = results.filter("towerId == \'{676A0286-C5CB-48B7-939C-EB9590740B24}\'").toPandas()

predict_fig = model.plot(test_pd, xlabel='date', ylabel='records')
 
# adjust figure to display dates from last year + the 90 day forecast
xlim = predict_fig.axes[0].get_xlim()
new_xlim = ( xlim[1]-(180.0+10.0), xlim[1]-10.0)
predict_fig.axes[0].set_xlim(new_xlim)
 
display(predict_fig)

# COMMAND ----------

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler



# COMMAND ----------

df = spark.sql("select CAST(datetime as date) as ds, totalRecords_CDR, towerId from geospatial_tomasz.CDR_day_gold")


# COMMAND ----------

df_pivot = df.groupBy("towerId").pivot("ds").sum("totalRecords_CDR")

display(df_pivot)

# COMMAND ----------

from datetime import date, timedelta

start_date = date(2021, 1, 1) 
end_date = date(2021, 12, 31)    # perhaps date.now()

delta = end_date - start_date   # returns timedelta

days = []
for i in range(delta.days + 1):
    day = start_date + timedelta(days=i)
    days.append(day.strftime("%Y-%m-%d"))

# COMMAND ----------

vecAssembler = VectorAssembler(inputCols=days, outputCol="features")
df_kmeans = vecAssembler.transform(df_pivot).select('towerId', 'features')
df_kmeans.show()

# COMMAND ----------

from pyspark.ml.evaluation import ClusteringEvaluator
# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# COMMAND ----------

df_kmeans

# COMMAND ----------

import numpy as np

cost = list()
for k in range(2,20):
    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
    model = kmeans.fit(df_kmeans.sample(False,0.1, seed=42))
    cost.append(model.summary.trainingCost)

# COMMAND ----------

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
plt.plot(range(2,15), cost, 'b*-')
plt.xlabel('Number of clusters');
plt.ylabel('Within Set Sum of Squared Error');
plt.title('Elbow for K-Means clustering');
# Uncomment the next line
display(fig)

# COMMAND ----------

print(cost)

# COMMAND ----------

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Loads data.
#dataset = spark.read.format("libsvm").load("data/mllib/sample_kmeans_data.txt")
vecAssembler = VectorAssembler(inputCols=days, outputCol="features")
df_kmeans = vecAssembler.transform(df_pivot).select('towerId', 'features')
df_kmeans.show()
# Trains a k-means model.
kmeans = KMeans().setK(2).setSeed(1)
model = kmeans.fit(df_kmeans)

# Make predictions
predictions = model.transform(df_kmeans)

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)
    
display(predictions)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT date_format(Datetime, 'E') , avg(totalRecords_CDR), towerId,
# MAGIC CASE WHEN date_format(Datetime, 'E') = 'Mon' THEN 1
# MAGIC      WHEN date_format(Datetime, 'E') = 'Tue' THEN 2
# MAGIC      WHEN date_format(Datetime, 'E') = 'Wed' THEN 3
# MAGIC      WHEN date_format(Datetime, 'E') = 'Thu' THEN 4
# MAGIC      WHEN date_format(Datetime, 'E') = 'Fri' THEN 5
# MAGIC      WHEN date_format(Datetime, 'E') = 'Sat' THEN 6
# MAGIC      ELSE 7
# MAGIC END AS weekday_num
# MAGIC FROM geospatial_tomasz.CDR_day_gold 
# MAGIC WHEN
# MAGIC GROUP BY date_format(Datetime, 'E'), towerId
# MAGIC ORDER BY weekday_num, towerId

# COMMAND ----------

""
