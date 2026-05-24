"""End-to-end ML pipeline: featurise -> train LR -> evaluate -> save.

A Pipeline groups every transformation + estimator into a single object.
After .fit() you get a PipelineModel that:
  - applies the same transformations to new data (no train/serve skew)
  - can be saved and loaded as a self-contained artifact
"""

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import (
    Imputer,
    OneHotEncoder,
    StringIndexer,
    VectorAssembler,
)
from pyspark.sql import SparkSession


def main() -> None:
    spark = (
        SparkSession.builder
        .appName("titanic-train")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    df = (
        spark.read.option("header", True).option("inferSchema", True)
        .csv("data/titanic.csv")
        .select("Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare")
    )

    train, test = df.randomSplit([0.8, 0.2], seed=42)
    print(f"train={train.count()}  test={test.count()}")

    sex_idx = StringIndexer(inputCol="Sex", outputCol="SexIdx", handleInvalid="keep")
    pclass_ohe = OneHotEncoder(inputCols=["Pclass"], outputCols=["PclassVec"])
    age_imp = Imputer(inputCols=["Age"], outputCols=["AgeImp"], strategy="mean")

    assembler = VectorAssembler(
        inputCols=["SexIdx", "PclassVec", "AgeImp", "SibSp", "Parch", "Fare"],
        outputCol="features",
    )

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="Survived",
        maxIter=20,
        regParam=0.0,
    )

    pipeline = Pipeline(stages=[sex_idx, pclass_ohe, age_imp, assembler, lr])

    model = pipeline.fit(train)
    preds = model.transform(test)

    auc = BinaryClassificationEvaluator(
        labelCol="Survived",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    ).evaluate(preds)
    print(f"\nholdout AUC = {auc:.3f}")

    preds.select("Survived", "prediction", "probability").show(10, truncate=False)

    out = "models/titanic_lr"
    model.write().overwrite().save(out)
    print(f"saved pipeline model to {out}")

    spark.stop()


if __name__ == "__main__":
    main()
