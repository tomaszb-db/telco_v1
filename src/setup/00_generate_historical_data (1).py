# Databricks notebook source
#table definitions from setup
db_name = "geospatial_tomasz"
cell_tower_table = "cell_tower_geojson"
area_codes_table = "area_codes" 
phone_numbers_table = "phone_numbers"

#historical tables
RSSI_table = "telco_signal_strength_hist"
CDR_table = "telco_CDR_hist"
PCMD_table = "telco_PCMD_hist"

# COMMAND ----------

#get tower IDs for data generation
import pyspark.sql.functions as F

df_towers = spark.sql("select * from {0}.{1}".format(db_name, cell_tower_table))

#filter on a specific provider
df_v_towers = df_towers.filter((F.lower(df_towers.properties.Licensee)).rlike("verizon"))

#get tower IDs in list format
globalIds = df_v_towers.select(df_v_towers.properties["GlobalID"]).rdd.flatMap(lambda x: x).collect()

#get subscriber ID to phone number mapping
phone_numbers_df = spark.sql("select * from {}.{}".format(db_name, phone_numbers_table))


# COMMAND ----------

#RSSI Data Stream Generation

from pyspark.sql.types import LongType, IntegerType, StringType, FloatType, DateType
from datetime import datetime, timedelta
import dbldatagen as dg
import dbldatagen.distributions as dist

#create data
partitions_requested = 36
data_rows = 1000000000

spark.conf.set("spark.sql.shuffle.partitions", partitions_requested)

now = datetime.now()
now_str = now.strftime("%Y-%m-%d %H:%M:%S")

df_spec_rssi = (dg.DataGenerator(spark, name="signalStrengthRecords", rows=data_rows, partitions=partitions_requested)
                            .withIdOutput()
                            .withColumn("towerId", StringType(), values=globalIds, random=True)
                            .withColumn("RSRP", FloatType(), minValue=-30, maxValue=-120, random=True, distribution=dist.Gamma(5.0,3.0))
                            .withColumn("RSRQ", FloatType(), minValue=-1, maxValue=-20.0, random=True, distribution=dist.Gamma(4.0, 2.0))
                            .withColumn("SINR", FloatType(), minValue=-5, maxValue=13, random=True, distribution=dist.Gamma(10.0, 1.0))
                            .withColumn("event_ts", "timestamp", begin="2021-01-01 00:00:00", end=now_str, interval="seconds=1", random=True)
                            )

dfTestData_rssi = df_spec_rssi.build()

#write out files to table
dfTestData_rssi.write.format("delta").mode("overwrite").saveAsTable("{}.{}".format(db_name, RSSI_table))

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select * from geospatial_tomasz.telco_signal_strength_hist

# COMMAND ----------

#CDR Historical Data Generation
  
#generate data
partitions_requested = 36
data_rows = 1000000000

now = datetime.now()
now_str = now.strftime("%Y-%m-%d %H:%M:%S")

df_spec_cdr = (dg.DataGenerator(spark, name="signalStrengthRecords", rows=data_rows, partitions=partitions_requested)
                            .withIdOutput()
                            .withColumn("towerId", StringType(), values=globalIds, random=True)
                            .withColumn("subscriberId", IntegerType(), minValue=1, maxValue=1000000, random=True)
                            .withColumn("otherId", IntegerType(), minValue=1, maxValue=1000000, random=True)
                            .withColumn("type", StringType(), values=["call", "text"], random=True, weights=[2,8])
                            .withColumn("status_hidden", StringType(), values=["answered", "missed", "dropped"], random=True, weights=[8,3,1])
                            .withColumn("duration_hidden", IntegerType(), minValue=1, maxValue=110, random=True, distribution=dist.Gamma(1, 1))
                            .withColumn("event_ts", "timestamp", begin="2021-01-01 00:00:00", end=now_str, interval="seconds=1", random=True)
                            )

df_TestData_cdr = df_spec_cdr.build()

#extra stransformations to add phone numbers from IDs

df_TestData_renamedId = df_TestData_cdr.withColumn("rId", F.col("Id")).drop("Id")

df_phoneJoinedData = df_TestData_renamedId \
      .join(phone_numbers_df.select(F.col("phone_number").alias("subscriber_phone"), F.col("id").alias("id_sub")), F.col("subscriberId") == F.col("id_sub")) \
      .join(phone_numbers_df.select(F.col("phone_number").alias("other_phone"), F.col("id").alias("id_other")), F.col("otherId") == F.col("id_other"))

df_withText = df_phoneJoinedData.withColumn("status", F.when(F.col("type") == "text", None).otherwise(F.col("status_hidden"))) \
    .drop("status_hidden")                                                                                       \
    .withColumn("duration", F.when(F.col("type") == "text", None).otherwise(F.col("duration_hidden")))           \
    .drop("duration_hidden")

df_withText.write.format("delta").mode("overwrite").saveAsTable("{}.{}".format(db_name, CDR_table))

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select * from geospatial_tomasz.telco_CDR_hist

# COMMAND ----------

#PCMD Sample Data
def foreach_batch_function_pcmd(df, epoch_id):
  df.coalesce(1)
  df.write.mode("append").json(PCMD_dir)
  
partitions_requested = 36
data_rows = 1000000000

now = datetime.now()
now_str = now.strftime("%Y-%m-%d %H:%M:%S")

df_spec_pcmd = (dg.DataGenerator(spark, name="pcmdSample", partitions=partitions_requested, verbose=True)
                           .withIdOutput()
                           .withColumn("towerId", StringType(), values=globalIds, random=True)
                           .withColumn("subscriberId", IntegerType(), minValue=1, maxValue=1000000, random=True)
                           .withColumn("ProcedureId", IntegerType(), values=[11, 15, 16], weights=[51, 5, 50])
                           .withColumn("ProcedureDuration", FloatType(), min=.0001, max=1)
                           .withColumn("event_ts", "timestamp", begin="2021-01-01 00:00:00", end=now_str, interval="seconds=1", random=True)
          )

df_TestData_pcmd = df_spec_pcmd.build()

df_TestData_pcmd.write.format("delta").mode("overwrite").saveAsTable("{}.{}".format(db_name, PCMD_table))

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select * from geospatial_tomasz.telco_PCMD_hist

# COMMAND ----------


