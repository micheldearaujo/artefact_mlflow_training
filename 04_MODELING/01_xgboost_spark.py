# Databricks notebook source
# MAGIC %md
# MAGIC ## Template
# MAGIC
# MAGIC **Objective**: This notebook's objective is to Train an XGBoost Regressor model from the Scikit-learn library to compare with the XGBoost from SparkML library.
# MAGIC
# MAGIC Objective: Regressor model to estimate the Fish's ``Weight``
# MAGIC
# MAGIC **Takeaways**: The key takeaways of this notebook are:
# MAGIC
# MAGIC -
# MAGIC -
# MAGIC -
# MAGIC -
# MAGIC -

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.0 Imports

# COMMAND ----------

# MAGIC %run ../01_CONFIG/utils

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.0 Data Loading

# COMMAND ----------

fishDF = spark.createDataFrame(pd.read_csv("../Fish.csv"))

# COMMAND ----------

display(fishDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.0 Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1 One Hot Encoding Get Dummies on the Species Column

# COMMAND ----------

from pyspark.ml.feature import StringIndexer


uniqueSpecies = fishDF.select("Species").distinct() # Use distinct values to demonstrate how StringIndexer works

indexer = StringIndexer(inputCol="Species", outputCol="species_index") # Set input column and new output column
indexerModel = indexer.fit(uniqueSpecies)                                  # Fit the indexer to learn room type/index pairs
indexedDF = indexerModel.transform(uniqueSpecies)                          # Append a new column with the index

display(indexedDF)

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder

encoder = OneHotEncoder(inputCols=["species_index"], outputCols=["encoded_species"])
encoderModel = encoder.fit(indexedDF)
encodedDF = encoderModel.transform(indexedDF)
display(encodedDF)

# COMMAND ----------

from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[
    indexer,
    encoder,
])

# COMMAND ----------

pipelineModel = pipeline.fit(fishDF)
featurizedDF = pipelineModel.transform(fishDF)

# COMMAND ----------

display(featurizedDF)

# COMMAND ----------

fishDF.columns

# COMMAND ----------


from pyspark.ml.feature import VectorAssembler

featureCol =  ['encoded_species', 'Length1', 'Length2', 'Length3', 'Height', 'Width']
assembler = VectorAssembler(inputCols=featureCol, outputCol="features")

featurizedDF = assembler.transform(featurizedDF)

display(featurizedDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.0 Train Model

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.1 Train test Split

# COMMAND ----------

trainDF, testDF = featurizedDF.randomSplit([0.8, 0.2])

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.2 Train the model

# COMMAND ----------

from xgboost.spark import SparkXGBRegressor

xgboost_on_spark = SparkXGBRegressor(
  features_col="features",
  label_col="Weight",
  num_workers=1,
)

# COMMAND ----------

xgboost_on_spark = xgboost_on_spark.fit(trainDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.0 Evaluate the model

# COMMAND ----------

testDFTransformed = xgboost_on_spark.transform(testDF)

# COMMAND ----------

display(testDFTransformed)

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

evaluator_rmse = RegressionEvaluator(predictionCol='prediction', labelCol='Weight', metricName='rmse')
evaluator_mae = RegressionEvaluator(predictionCol='prediction', labelCol='Weight', metricName='mae')
evaluator_mse = RegressionEvaluator(predictionCol='prediction', labelCol='Weight', metricName='mse')

testError_rmse = evaluator_rmse.evaluate(testDFTransformed)
testError_mae = evaluator_mae.evaluate(testDFTransformed)
testError_mse = evaluator_mse.evaluate(testDFTransformed)

print("MAE: {}".format(testError_rmse))
print("RMSE: {}".format(testError_mae))
print("MSE: {}".format(testError_mse))

# COMMAND ----------

testingDF = testDFTransformed.toPandas()
testingDF["Residual"] = testingDF["Weight"] - testingDF["prediction"]

# COMMAND ----------

testingDF.head()

# COMMAND ----------

sns.scatterplot(
    data=testingDF,
    x='Weight',
    y='prediction',
)

# COMMAND ----------

sns.histplot(
    data=testingDF,
    x='Residual',
    kde=True
)
plt.show()

# COMMAND ----------


