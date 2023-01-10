# Databricks notebook source
#table definitions from setup
db_name = "geospatial_tomasz"
cell_tower_table = "cell_tower_geojson"
area_codes_table = "area_codes" 
phone_numbers_table = "phone_numbers"

#historical tables
CDR_table = "telco_CDR_ml"


# COMMAND ----------

import pyspark.sql.functions as F

df_towers = spark.sql("select * from {0}.{1}".format(db_name, cell_tower_table))

#filter on a specific provider
df_v_towers = df_towers.filter((F.lower(df_towers.properties.Licensee)).rlike("verizon"))

# COMMAND ----------

cities = ["Denver", "Boulder"]
df_v_towers_denver_boulder = df_v_towers.filter(df_towers.properties.LocCity.isin(cities))

globalIds = df_v_towers_denver_boulder.select(df_v_towers.properties["GlobalID"]).rdd.flatMap(lambda x: x).collect()

#get subscriber ID to phone number mapping
phone_numbers_df = spark.sql("select * from {}.{}".format(db_name, phone_numbers_table))

# COMMAND ----------

from datetime import date, timedelta

#CDR Historical Data Generation
  
#generate data
partitions_requested = 36
data_rows = 1000000000

#hour distributions
hours = [x for x in range(0, 24)]
busier_hours = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
less_busy_hours = [1, 2, 3, 4, 5, 6, 7, 8, 22, 23, 24]

#weekdays vs weekends
weekends = []
weekdays = []

def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)

for dt in daterange(date(2021, 1, 1), date(2021, 12, 31)):
  if dt.weekday() in [0, 1, 2, 3, 4]:
    weekdays.append(dt.strftime("%Y-%m-%d"))
  else:
    weekends.append(dt.strftime("%Y-%m-%d"))
    

all_days = weekends + weekdays
print(all_days)

# COMMAND ----------

#create baseline data
from pyspark.sql.types import LongType, IntegerType, StringType, FloatType, DateType
import dbldatagen as dg
import dbldatagen.distributions as dist
  
#generate data
partitions_requested = 20
data_rows = round(2920000000/2)


#generate baseline data
df_spec_cdr = (dg.DataGenerator(spark, name="baseline_CDR", rows=data_rows, partitions=partitions_requested)
                            .withIdOutput()
                            .withColumn("towerId", StringType(), values=globalIds, random=True)
                            .withColumn("subscriberId", IntegerType(), minValue=1, maxValue=1000000, random=True)
                            .withColumn("otherId", IntegerType(), minValue=1, maxValue=1000000, random=True)
                            .withColumn("type", StringType(), values=["call", "text"], random=True, weights=[2,8])
                            .withColumn("status_hidden", StringType(), values=["answered", "missed", "dropped"], random=True, weights=[8,3,1])
                            .withColumn("duration_hidden", IntegerType(), minValue=1, maxValue=110, random=True, distribution=dist.Gamma(1, 1))
                            .withColumn("hours_hidden", IntegerType(), values=hours, random=True, omit=True)
                            .withColumn("minutes_hidden", IntegerType(), minValue=0, maxValue=59, random=True, omit=True)
                            .withColumn("seconds_hidden", IntegerType(), minValue=0, maxValue=59, random=True, omit=True)
                            .withColumn("fulltime_hidden", StringType(), expr="concat(hours_hidden, ':', minutes_hidden, ':', seconds_hidden)", baseColumn=["hours_hidden", "minutes_hidden", "seconds_hidden"], omit=True)
                            .withColumn("date_hidden", StringType(), values=all_days, random=True, omit=True)
                            .withColumn("event_ts", TimestampType(), expr="to_timestamp(concat(date_hidden, ' ', fulltime_hidden))", baseColumn=["date_hidden", "fulltime_hidden"])
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

#create create weekend data
from pyspark.sql.types import LongType, IntegerType, StringType, FloatType, DateType, TimestampType
import dbldatagen as dg
import dbldatagen.distributions as dist
  
#generate data
partitions_requested = 20
data_rows = round(2920000000/7)

#generate baseline data
df_spec_cdr = (dg.DataGenerator(spark, name="weekend_CDR", rows=data_rows, partitions=partitions_requested)
                            .withIdOutput()
                            .withColumn("towerId", StringType(), values=globalIds, random=True, weights=[2, 2, 7, 2, 8, 2])
                            .withColumn("subscriberId", IntegerType(), minValue=1, maxValue=1000000, random=True)
                            .withColumn("otherId", IntegerType(), minValue=1, maxValue=1000000, random=True)
                            .withColumn("type", StringType(), values=["call", "text"], random=True, weights=[2,8])
                            .withColumn("status_hidden", StringType(), values=["answered", "missed", "dropped"], random=True, weights=[30,16,1])
                            .withColumn("duration_hidden", IntegerType(), minValue=1, maxValue=110, random=True, distribution=dist.Gamma(1, 1))
                            .withColumn("hours_hidden", IntegerType(), values=hours, random=True, weights=[1, 1, 1, 1, 1, 1, 2, 3, 3, 6, 9, 7, 9, 7, 7, 8, 7, 8, 5, 6, 7, 8, 7, 6], omit=True)
                            .withColumn("minutes_hidden", IntegerType(), minValue=0, maxValue=59, random=True, omit=True)
                            .withColumn("seconds_hidden", IntegerType(), minValue=0, maxValue=59, random=True, omit=True)
                            .withColumn("fulltime_hidden", StringType(), expr="concat(hours_hidden, ':', minutes_hidden, ':', seconds_hidden)", baseColumn=["hours_hidden", "minutes_hidden", "seconds_hidden"], omit=True)
                            .withColumn("date_hidden", StringType(), values=weekends, random=True, omit=True)
                            .withColumn("event_ts", TimestampType(), expr="to_timestamp(concat(date_hidden, ' ', fulltime_hidden))", baseColumn=["date_hidden", "fulltime_hidden"])
                            )

busier_hours = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
less_busy_hours = [1, 2, 3, 4, 5, 6, 7, 8, 22, 23, 24]
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

df_withText.write.format("delta").mode("append").saveAsTable("{}.{}".format(db_name, CDR_table))

# COMMAND ----------

#create create weekday data
from pyspark.sql.types import LongType, IntegerType, StringType, FloatType, DateType, TimestampType
import dbldatagen as dg
import dbldatagen.distributions as dist
  
#generate data
partitions_requested = 20
data_rows = round(2920000000/2)

#generate baseline data
df_spec_cdr = (dg.DataGenerator(spark, name="weekend_CDR", rows=data_rows, partitions=partitions_requested)
                            .withIdOutput()
                            .withColumn("towerId", StringType(), values=globalIds, random=True, weights=[10, 9, 3, 12, 3, 11])
                            .withColumn("subscriberId", IntegerType(), minValue=1, maxValue=1000000, random=True)
                            .withColumn("otherId", IntegerType(), minValue=1, maxValue=1000000, random=True)
                            .withColumn("type", StringType(), values=["call", "text"], random=True, weights=[2,8])
                            .withColumn("status_hidden", StringType(), values=["answered", "missed", "dropped"], random=True, weights=[30,16,1])
                            .withColumn("duration_hidden", IntegerType(), minValue=1, maxValue=110, random=True, distribution=dist.Gamma(1, 1))
                            .withColumn("hours_hidden", IntegerType(), values=hours, random=True, weights=[1, 1, 1, 1, 1, 1, 2, 3, 3, 6, 9, 7, 9, 7, 7, 8, 7, 8, 5, 6, 7, 8, 7, 6], omit=True)
                            .withColumn("minutes_hidden", IntegerType(), minValue=0, maxValue=59, random=True, omit=True)
                            .withColumn("seconds_hidden", IntegerType(), minValue=0, maxValue=59, random=True, omit=True)
                            .withColumn("fulltime_hidden", StringType(), expr="concat(hours_hidden, ':', minutes_hidden, ':', seconds_hidden)", baseColumn=["hours_hidden", "minutes_hidden", "seconds_hidden"], omit=True)
                            .withColumn("date_hidden", StringType(), values=weekdays, random=True, omit=True)
                            .withColumn("event_ts", TimestampType(), expr="to_timestamp(concat(date_hidden, ' ', fulltime_hidden))", baseColumn=["date_hidden", "fulltime_hidden"])
                            )

busier_hours = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
less_busy_hours = [1, 2, 3, 4, 5, 6, 7, 8, 22, 23, 24]
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

df_withText.write.format("delta").mode("append").saveAsTable("{}.{}".format(db_name, CDR_table))

# COMMAND ----------

import pyspark.sql.functions as F

def createWindowTable(sourceTable, targetTable, windowTime, towersTable):
  source_df = spark.sql("select * from {}".format(sourceTable))
  
  grouped_df = source_df.groupBy(F.window("event_ts", windowTime), "towerId")\
                        .agg(F.count(F.when(F.col("status") == "dropped", True)).alias("dropped"),   \
                          F.count(F.when(F.col("status") == "answered", True)).alias("answered"), \
                          F.count(F.when(F.col("status") == "missed", True)).alias("missed"),     \
                          F.count(F.when(F.col("type") == "text", True)).alias("text"),           \
                          F.count(F.when(F.col("type") == "call", True)).alias("call"),           \
                          F.count(F.lit(1)).alias("totalRecords_CDR"),                            \
                          F.first("window.start").alias("window_start"))                         \
                          .withColumn("datetime", (F.col("window_start")))
  
  df_towers = spark.sql("select * from {}".format(towersTable))
  df_towers_trunc = df_towers.select(df_towers.properties.GlobalID.alias("GlobalId"), df_towers.properties.LocCity.alias("City"), df_towers.properties.LocCounty.alias("County"), df_towers.properties.LocState.alias("State"), df_towers.geometry.coordinates[0].alias("Longitude"), df_towers.geometry.coordinates[1].alias("Latitude"))
  
  grouped_df_with_tower = grouped_df.join(df_towers_trunc, grouped_df.towerId == df_towers_trunc.GlobalId).drop("GlobalId")
  
  grouped_df_with_tower_ordered = grouped_df_with_tower.select("datetime",     \
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
  
  grouped_df_with_tower_ordered.write.format("delta").mode("overwrite").saveAsTable("{}".format(targetTable))

# COMMAND ----------

#hour table 
sourceTable = "{}.{}".format(db_name, CDR_table_ml)
targetTable = "{}.{}".format(db_name, CDR_table_hour_ml)
windowTime = "1 hour"
towersTable = "{}.{}".format(db_name, cell_tower_table)
createWindowTable(sourceTable, targetTable, windowTime, towersTable)

# COMMAND ----------

#day table
sourceTable = "{}.{}".format(db_name, CDR_table)
targetTable = "{}.{}".format(db_name, CDR_table_day_ml)
windowTime = "1 day"
towersTable = "{}.{}".format(db_name, cell_tower_table)
createWindowTable(sourceTable, targetTable, windowTime, towersTable)
