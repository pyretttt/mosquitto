"""Load CSV, inspect, run a few transformations, look at the query plan."""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def main() -> None:
    spark = (
        SparkSession.builder
        .appName("titanic-explore")
        .master("local[*]")
        # 4 partitions for shuffles is sane for a 100-row dataset.
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv("data/titanic.csv")
    )

    print("\n--- schema ---")
    df.printSchema()

    print(f"\nrows = {df.count()}")

    print("\n--- survival rate per class & sex ---")
    (
        df.groupBy("Pclass", "Sex")
          .agg(F.avg("Survived").alias("survival_rate"),
               F.count("*").alias("n"))
          .orderBy("Pclass", "Sex")
          .show()
    )

    print("\n--- query plan for: filter + select + groupBy ---")
    plan_df = (
        df.filter(F.col("Age").isNotNull())
          .select("Pclass", "Sex", "Age", "Survived")
          .groupBy("Pclass")
          .agg(F.avg("Age").alias("avg_age"))
    )
    plan_df.explain(True)
    plan_df.show()

    print("default partitions on initial CSV read:", df.rdd.getNumPartitions())
    spark.stop()


if __name__ == "__main__":
    main()
