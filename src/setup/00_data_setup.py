# Databricks notebook source
#table definitions
db_name = "geospatial_tomasz"
cell_tower_table = "cell_tower_geojson"
area_codes_table = "area_codes"
phone_numbers_table = "phone_numbers"

# COMMAND ----------

#get and create tower data table

# COMMAND ----------

#get and create areacodes table

# COMMAND ----------

#create phone numbers table
import random
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import dbldatagen as dg

#random numbers with real area codes UDF
area_codes_table = spark.sql("select * from {0}.{1}".format(db_name, area_codes_table))
area_codes = area_codes_table.select("AreaCode").rdd.flatMap(lambda x: x).collect()

df_phoneNums = (dg.DataGenerator(sparkSession=spark, name="phone_numbers", rows=1000000, partitions=4, randomSeedMethod="hash_fieldname")
                .withColumn("phoneLastDigits", template='ddd-dddd', omit=True)
                .withColumn("areaCode",  StringType(), values=area_codes, random=True, omit=True)
                .withColumn("phone_number", StringType(), expr="concat(areaCode, '-', phoneLastDigits)", baseColumn=["areaCode", "phoneLastDigits"])
               )

df_phoneData = df_phoneNums.build()

df_phoneData_withId = df_phoneData.withColumn("id", F.monotonically_increasing_id())
df_phoneData_withId.write.mode("overwrite").saveAsTable("{0}.{1}".format(db_name, phone_numbers_table))
