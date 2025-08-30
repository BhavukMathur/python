from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# 1. Initialize Spark Session
spark = SparkSession.builder \
    .appName("Simple PySpark Example") \
    .getOrCreate()

# 2. Create sample data
data = [
    (1, "Alice", 29),
    (2, "Bob", 31),
    (3, "Charlie", 25),
    (4, "David", 40)
]

columns = ["id", "name", "age"]

# 3. Create DataFrame
df = spark.createDataFrame(data, columns)

print("=== Original DataFrame ===")
df.show()

# 4. Select & filter
print("=== People older than 30 ===")
df.filter(col("age") > 30).show()

# 5. Transform (add new column)
print("=== Add 10 years to age ===")
df.withColumn("age_plus_10", col("age") + 10).show()

# 6. Group by
print("=== Average age ===")
df.groupBy().avg("age").show()

# 7. Stop Spark
spark.stop()
