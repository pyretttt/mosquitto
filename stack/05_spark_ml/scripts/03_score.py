"""Load the saved PipelineModel and score new rows.

Proves the artifact is self-contained: same indexers, same imputer means,
same encoder vocab — no train/serve skew.
"""

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession


def main() -> None:
    spark = (
        SparkSession.builder.appName("titanic-score").master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    model = PipelineModel.load("models/titanic_lr")

    new = spark.createDataFrame(
        [
            (3, "male",   28.0, 0, 0, 7.25),    # young 3rd class man -> low
            (1, "female", 35.0, 1, 0, 71.0),    # rich 1st class woman -> high
            (2, "female", 10.0, 0, 1, 30.0),    # 2nd class girl -> high
        ],
        ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"],
    )

    preds = model.transform(new)
    preds.select("Pclass", "Sex", "Age", "prediction", "probability").show(truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
