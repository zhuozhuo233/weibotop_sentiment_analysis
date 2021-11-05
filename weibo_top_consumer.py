import sys
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

from IPython.display import clear_output
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import time


@udf(returnType=StringType())
def gettitle(column):
    jsonobject = json.loads(column)
    jsonobject = json.loads(jsonobject)
    if "title" in jsonobject:
        return str(jsonobject['title'])
    return ""


@udf(returnType=DoubleType())
def getscore(column):
    jsonobject = json.loads(column)
    jsonobject = json.loads(jsonobject)
    if "sentiment_score" in jsonobject:
        return float(jsonobject['sentiment_score'])
    return 0.5


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("参数错误", file=sys.stderr)
        sys.exit(-1)
    spark = SparkSession\
        .builder\
        .appName("WeiboSpark")\
        .getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    bootstrapServers = sys.argv[1]
    subscribeType = sys.argv[2]
    topics = sys.argv[3]
    lines = spark\
        .readStream\
        .format("kafka")\
        .option("kafka.bootstrap.servers", bootstrapServers)\
        .option(subscribeType, topics)\
        .load()
    kafka_value_tb = lines.selectExpr("CAST(value AS STRING) as json", "timestamp")
    weibo_table = kafka_value_tb.select(gettitle(col("json")).alias("title"),getscore(col("json")).alias("sentiment_score"), col("timestamp"))
    stat_avg = weibo_table.groupBy(window(col("timestamp"), "30 seconds", "10 seconds"),col("title")).avg("sentiment_score").where("unix_timestamp(window.end)=int(unix_timestamp(current_timestamp)/10)*10")
    queryStream = (stat_avg.writeStream.format("memory").queryName("weibotop").outputMode("complete").start())
    try:
        i = 1
        while True:
            print("count", str(i))
            df = spark.sql("""select * from weibotop""").toPandas()
            print(df)
            df.to_csv("weibo_sentiment_result.csv")

            time.sleep(10)
            i = i + 1
    except KeyboardInterrupt:
        print("process interrupted")
    queryStream.awaitTermination()
