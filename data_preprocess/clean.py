import pyspark

from pyspark.sql import SQLContext
sc = pyspark.SparkContext.getOrCreate()
sqlContext = SQLContext(sc)

# read data
df = sc.textFile('s3://yelpdataset-cc2018/yelp_academic_dataset_review.csv').map(lambda x: x.split(","))

df = df.map(lambda l: (l[8], l[0], l[5]))



