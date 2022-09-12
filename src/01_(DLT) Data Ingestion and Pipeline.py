# Databricks notebook source
# MAGIC %md
# MAGIC ## Telecommunications Reliability Metrics
# MAGIC 
# MAGIC **Telecommunications LTE Architecture**
# MAGIC <br>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC The modern telecommunications network consists of the **eNodeB (Evolved Node B)** is the hardware that communicates directly with the **UE (User Enitity such as a Mobile Phone)**. The **MME (Mobility Management Entity)** manages the entire process from a cell phone making a connection to a network to a paging message being sent to the mobile phone.  
# MAGIC 
# MAGIC <img style="margin-left: 500px" src="https://raw.githubusercontent.com/tomaszb-db/telco_v0/master/Images/LTE_architecture.png" width="700"/>
# MAGIC 
# MAGIC **Use Case Overview**
# MAGIC * Telecommunications services collect many different forms of data to observe overall network reliability as well as to predict how best to expand the network to reach more customers. Some typical types of data collected are:
# MAGIC   - **PCMD (Per Call Measurement Data):** granular details of all network processes as MME (Mobility Management Entity) manages processes between the UE and the rest of the network
# MAGIC   - **CDR (Call Detail Records):** high level data describing call and SMS activity with fields such as phone number origin, phone number target, status of call/sms, duration, etc. 
# MAGIC * This data can be collected and used in provide a full view of the health of each cell tower in the network as well as the network as a whole. 
# MAGIC * **Note:** for this demo we will be primarily focused on CDR data but will also have a small sample of what PCMD could look like.
# MAGIC 
# MAGIC **Business Impact of Solution**
# MAGIC * **Ease of Scaling:** with large amounts of data being generated by a telecommunications system, Databricks can provide the ability to scale so that the data can be reliably ingested and analyzed.  
# MAGIC * **Greater Network Reliability:** with the ability to monitor and predict dropped communications and more generally network faults, telecommunications providers can ultimately deliver better service for their customers and reduce churn.
# MAGIC 
# MAGIC **Full Architecture from Ingestion to Analytics and Machine Learning**
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/tomaszb-db/telco_v1/master/Images/Telco_Full_v2.png" width="1000"/>

# COMMAND ----------

import dlt
import pyspark.sql.functions as F
from pyspark.sql.types import *

#table location definitions
db_name = "geospatial_tomasz"
cell_tower_table = "cell_tower_geojson"

#locations of data streams
CDR_dir = "dbfs:/tmp/tomasz.bacewicz@databricks.com/telco_CDR"
RSSI_dir = "dbfs:/tmp/tomasz.bacewicz@databricks.com/telco_signal_strength"
PCMD_dir = "dbfs:/tmp/tomasz.bacewicz@databricks.com/telco_PCMD"

CDR_schema = "dbfs:/tmp/tomasz.bacewicz@databricks.com/CDR_schema/"
RSSI_schema = "dbfs:/tmp/tomasz.bacewicz@databricks.com/RSSI_schema/"
PCMD_schema = "dbfs:/tmp/tomasz.bacewicz@databricks.com/PCMD_schema/"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Ingestion with Delta Live Tables
# MAGIC 
# MAGIC To simplify the ingestion process and accelerate our developments, we'll leverage Delta Live Table (DLT).
# MAGIC 
# MAGIC DLT let you declare your transformations and will handle the Data Engineering complexity for you:
# MAGIC - Data quality tracking with expectations
# MAGIC - Continuous or scheduled ingestion, orchestrated as pipeline
# MAGIC - Build lineage and manage data dependencies
# MAGIC - Automating scaling and fault tolerance
# MAGIC 
# MAGIC **Bronze Layer** 
# MAGIC 
# MAGIC * Ingestion here starts with loading CDR and PCMD data directly from S3 using Autoloader. Though in this example JSON files are loaded into S3 from where Autoloader will then ingest these files into the bronze layer, streams from Kafka, Kinesis, etc. are supported by simply changing the "format" option on the read operation.
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/tomaszb-db/telco_v1/master/Images/Telco_Bronze_v2.png?token=GHSAT0AAAAAABRG5BOEJYYCPT4PZ36K2QY2YYE6N4A" width="1000"/>

# COMMAND ----------

@dlt.table(comment="CDR Stream - Bronze")
def cdr_stream_bronze():
  return spark.readStream.format("cloudFiles")  \
                          .option("cloudFiles.format", 'json') \
                          .option('header', 'false')  \
                          .option("mergeSchema", "true")         \
                          .option("cloudFiles.inferColumnTypes", "true") \
                          .option("cloudFiles.schemaLocation", CDR_schema) \
                          .load(CDR_dir)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM geospatial_tomasz.cdr_stream_bronze_static

# COMMAND ----------

@dlt.table(comment="RSSI Stream - Bronze")
def rssi_stream_bronze():
  return spark.readStream.format("cloudFiles")  \
          .option("cloudFiles.format", 'json')  \
          .option('header', 'false') \
          .option("mergeSchema", "true")         \
          .option("cloudFiles.inferColumnTypes", "true") \
          .option("cloudFiles.schemaLocation", RSSI_schema) \
          .load(RSSI_dir)

# COMMAND ----------

@dlt.table(comment="PCMD Stream - Bronze")
def pcmd_stream_bronze():
  return spark.readStream.format("cloudFiles")  \
                          .option("cloudFiles.format", 'json') \
                          .option('header', 'false')  \
                          .option("mergeSchema", "true")         \
                          .option("cloudFiles.inferColumnTypes", "true") \
                          .option("cloudFiles.schemaLocation", PCMD_schema) \
                          .load(PCMD_dir)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM geospatial_tomasz.pcmd_stream_bronze_static

# COMMAND ----------

# MAGIC %md
# MAGIC ## Joining with Tower Data and Creating the Silver Layer 
# MAGIC 
# MAGIC **Silver Layer**
# MAGIC 
# MAGIC * In the silver layer, the data is refined removing nulls and duplicates while also joining tower information such as state, longitude, and latitude to allow for geospatial analysis. Stream-static joins are performed to do this with the streaming CDR and PCMD records being joined with static tower information which has been stored previously.
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/tomaszb-db/telco_v1/master/Images/Telco_Silver_v2.png" width="1000"/>

# COMMAND ----------

@dlt.view
def static_tower_data():
  df_towers = spark.sql("select * from {0}.{1}".format(db_name, cell_tower_table))
  
  return df_towers.select(df_towers.properties.GlobalID.alias("GlobalID"), df_towers.properties.LocCity.alias("City"), df_towers.properties.LocCounty.alias("County"), df_towers.properties.LocState.alias("State"), df_towers.geometry.coordinates[0].alias("Longitude"), df_towers.geometry.coordinates[1].alias("Latitude"))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM geospatial_tomasz.cell_tower_geojson

# COMMAND ----------

@dlt.table(comment="CDR Stream - Silver (Tower Info Added)")
def cdr_stream_silver():
  #get static tower data
  df_towers = dlt.read("static_tower_data")
  
  df_cdr_bronze = dlt.read_stream("cdr_stream_bronze")
  df_cdr_bronze_col_rename = df_cdr_bronze.withColumn("typeC", F.col("type")).drop("type")
  return df_cdr_bronze_col_rename.join(df_towers, df_cdr_bronze_col_rename.towerId == df_towers.GlobalID)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM geospatial_tomasz.cdr_stream_silver_static

# COMMAND ----------

@dlt.table(comment="RSSI Stream - Silver (Tower Info Added)")
def rssi_stream_silver():
  #get static tower data
  df_towers = dlt.read("static_tower_data")
  
  df_rssi_bronze = dlt.read_stream("rssi_stream_bronze")
  return df_rssi_bronze.join(df_towers, df_rssi_bronze.towerId == df_towers.GlobalID)

# COMMAND ----------

@dlt.table(comment="PCMD Stream - Silver (Tower Info Added)")
def pcmd_stream_silver():
  #get static tower data
  df_towers = dlt.read("static_tower_data")
  
  df_pcmd_bronze = dlt.read_stream("pcmd_stream_bronze")
  return df_pcmd_bronze.join(df_towers, df_pcmd_bronze.towerId == df_towers.GlobalID)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM geospatial_tomasz.pcmd_stream_silver_static

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregating on Various Time Periods to Create the Gold Layer
# MAGIC With Spark **Structured Streaming** the streaming records can be automatically aggregated with stateful processing. Here the aggregation is done on 1 minute intervals and the KPIs are aggregated accordingly. Any interval can be selected here and larger time window aggregations can be done on a scheduled basis with Databricks **Workflows**. For example, the records that are aggregated here at 1 minute intervals can then be aggregated to hour long intervals with a workflow that runs every hour. 
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/tomaszb-db/telco_v1/master/Images/Telco_Gold_v2.png" width="1000"/>

# COMMAND ----------

@dlt.table(comment="Aggregate CDR Stream - Gold (by Minute)")
def cdr_stream_minute_gold():
  df_cdr_silver = dlt.read_stream("cdr_stream_silver")
  
  
  #add widget to choose time window
  df_cdr_pivot_on_status_grouped_tower = df_cdr_silver \
                                                    .groupBy(F.window("event_ts", "1 minute"), "towerId")\
                                                    .agg(F.count(F.when(F.col("status") == "dropped", True)).alias("dropped"),   \
                                                    F.count(F.when(F.col("status") == "answered", True)).alias("answered"), \
                                                    F.count(F.when(F.col("status") == "missed", True)).alias("missed"),     \
                                                    F.count(F.when(F.col("typeC") == "text", True)).alias("text"),           \
                                                    F.count(F.when(F.col("typeC") == "call", True)).alias("call"),           \
                                                    F.count(F.lit(1)).alias("totalRecords_CDR"),                            \
                                                    F.first("window.start").alias("window_start"),                           \
                                                    F.first("Longitude").alias("Longitude"),                                 \
                                                    F.first("Latitude").alias("Latitude"),                                   \
                                                    F.first("City").alias("City"),                                           \
                                                    F.first("County").alias("County"),                                       \
                                                    F.first("State").alias("state"))                                        \
                                                    .withColumn("date", F.col("window_start"))                              
  


  
  df_cdr_pivot_on_status_grouped_tower_ordered = df_cdr_pivot_on_status_grouped_tower.select("date",     \
                                                                                             "towerId",  \
                                                                                             "answered", \
                                                                                             "dropped",  \
                                                                                             "missed",   \
                                                                                             "call",     \
                                                                                             "text",     \
                                                                                             "totalRecords_CDR", \
                                                                                             "Latitude", \
                                                                                             "Longitude",\
                                                                                             "City",     \
                                                                                             "County",   \
                                                                                             "State")
  
  return df_cdr_pivot_on_status_grouped_tower_ordered


# COMMAND ----------

@dlt.table(comment="Aggregate RSSI Stream - Gold (by Minute)")
def RSSI_stream_minute_gold():
  df_rssi_silver = dlt.read_stream("rssi_stream_silver")
  
  df_rssi_pivot_on_status_grouped_tower = df_rssi_silver  \
                                                    .groupBy(F.window("event_ts", "1 minute"), "towerId") \
                                                    .agg(F.avg(F.col("RSRP")).alias("avg_RSRP"), \
                                                    F.avg(F.col("RSRQ")).alias("avg_RSRQ"), \
                                                    F.avg(F.col("SINR")).alias("avg_SINR"), \
                                                    F.max(F.col("RSRP")).alias("max_RSRP"), \
                                                    F.max(F.col("RSRQ")).alias("max_RSRQ"), \
                                                    F.max(F.col("SINR")).alias("max_SINR"), \
                                                    F.min(F.col("RSRP")).alias("min_RSRP"), \
                                                    F.min(F.col("RSRQ")).alias("min_RSRQ"), \
                                                    F.min(F.col("SINR")).alias("min_SINR"), \
                                                    F.count(F.lit(1)).alias("totalRecords_RSSI"),
                                                    F.first("window.start").alias("window_start"), \
                                                    F.first("Longitude").alias("Longitude"),       \
                                                    F.first("Latitude").alias("Latitude"),         \
                                                    F.first("City").alias("City"),                 \
                                                    F.first("County").alias("County"),             \
                                                    F.first("State").alias("state"))              \
                                                    .withColumn("date", F.col("window_start"))
  
  
  df_rssi_pivot_on_status_grouped_tower_ordered = df_rssi_pivot_on_status_grouped_tower.select("date",  \
                                                                                        "towerId", 
                                                                                        "avg_RSRP", \
                                                                                        "avg_RSRQ", \
                                                                                        "avg_SINR", \
                                                                                        "max_RSRP", \
                                                                                        "max_RSRQ", \
                                                                                        "max_SINR", \
                                                                                        "min_RSRP", \
                                                                                        "min_RSRQ", \
                                                                                        "min_SINR", \
                                                                                        "totalRecords_RSSI", \
                                                                                        "Latitude", \
                                                                                        "Longitude",\
                                                                                        "City",     \
                                                                                        "County",   \
                                                                                        "State")
                                                                                      
  
  return df_rssi_pivot_on_status_grouped_tower_ordered

# COMMAND ----------

@dlt.table(comment="Aggregate PCMD Stream - Gold (by Minute)")
def PCMD_stream_minute_gold():
  df_pcmd_silver = dlt.read_stream("pcmd_stream_silver")
  
  #clean up the code
  df_pcmd_pivot_on_status_grouped_tower = df_pcmd_silver \
                                                    .groupBy(F.window("event_ts", "1 minute"), "towerId") \
                                                    .agg(F.avg(F.when(F.col("ProcedureId") == "11", F.col("ProcedureDuration"))).alias("avg_dur_request_to_release_bearer"),  \
                                                    F.avg(F.when(F.col("ProcedureId") == "15", F.col("ProcedureDuration"))).alias("avg_dur_notification_data_sent_to_UE"), \
                                                    F.avg(F.when(F.col("ProcedureId") == "16", F.col("ProcedureDuration"))).alias("avg_dur_request_to_setup_bearer"), \
                                                    F.max(F.when(F.col("ProcedureId") == "11", F.col("ProcedureDuration"))).alias("max_dur_request_to_release_bearer"),  \
                                                    F.max(F.when(F.col("ProcedureId") == "15", F.col("ProcedureDuration"))).alias("max_dur_notification_data_sent_to_UE"), \
                                                    F.max(F.when(F.col("ProcedureId") == "16", F.col("ProcedureDuration"))).alias("max_dur_request_to_setup_bearer"), \
                                                    F.min(F.when(F.col("ProcedureId") == "11", F.col("ProcedureDuration"))).alias("min_dur_request_to_release_bearer"),  \
                                                    F.min(F.when(F.col("ProcedureId") == "15", F.col("ProcedureDuration"))).alias("min_dur_notification_data_sent_to_UE"), \
                                                    F.min(F.when(F.col("ProcedureId") == "16", F.col("ProcedureDuration"))).alias("min_dur_request_to_setup_bearer"), \
                                                    F.count(F.lit(1)).alias("totalRecords_PCMD"), \
                                                    F.first("window.start").alias("window_start"), \
                                                    F.first("Longitude").alias("Longitude"),       \
                                                    F.first("Latitude").alias("Latitude"),         \
                                                    F.first("City").alias("City"),                 \
                                                    F.first("County").alias("County"),             \
                                                    F.first("State").alias("state"))              \
                                                    .withColumn("date", F.col("window_start"))
  
  df_pcmd_pivot_on_status_grouped_tower_ordered = df_pcmd_pivot_on_status_grouped_tower.select("date",  \
                                                                                        "towerId", \
                                                                                        "avg_dur_request_to_release_bearer", \
                                                                                        "avg_dur_notification_data_sent_to_UE", \
                                                                                        "avg_dur_request_to_setup_bearer", \
                                                                                        "max_dur_request_to_release_bearer", \
                                                                                        "max_dur_notification_data_sent_to_UE", \
                                                                                        "max_dur_request_to_setup_bearer", \
                                                                                        "min_dur_request_to_release_bearer", \
                                                                                        "min_dur_notification_data_sent_to_UE", \
                                                                                        "min_dur_request_to_setup_bearer", \
                                                                                        "totalRecords_PCMD", \
                                                                                        "Latitude", \
                                                                                        "Longitude",\
                                                                                        "City",     \
                                                                                        "County",   \
                                                                                        "State")
  
  return df_pcmd_pivot_on_status_grouped_tower_ordered

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregating on Larger Time Windows Through Scheduled Batch Workflows
# MAGIC Though not shown in this notebook, the aggregations created on the minute level can now be aggregated to an hourly or daily basis. Higher level aggregations such as hourly make it easier for data scientists to train machine learning models with meaningful data. Daily aggregations can be valueable to analysts who need to observe the historical trends of network performance.

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Using Databricks SQL for Analytics and Reliability Monitoring
# MAGIC Once the gold tables of different time aggregations are in place, analysis can be done in the Databricks SQL persona view. <a href="https://e2-demo-field-eng.cloud.databricks.com/sql/dashboards/b9a70988-9ed2-4fc0-b4f0-02a485b8e364-telco-cdr-overview?o=1444828305810485">Here</a> is a sample dashboard which shows the overall health of the network starting with the total number of calls and dropped calls, a geospatial view to analyze the problem areas, and lastly a table to see the find the details of every tower. 
# MAGIC 
# MAGIC <br>
# MAGIC <br>
# MAGIC 
# MAGIC <img src="https://raw.githubusercontent.com/tomaszb-db/telco_v0/master/Images/Telcodashboard.png" width="1000"/>