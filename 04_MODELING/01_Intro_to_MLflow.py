# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow
# MAGIC 
# MAGIC <a href="https://mlflow.org/docs/latest/concepts.html" target="_blank">MLflow</a> seeks to address these three core issues:
# MAGIC 
# MAGIC * It’s difficult to keep track of experiments
# MAGIC * It’s difficult to reproduce code
# MAGIC * There’s no standard way to package and deploy models
# MAGIC 
# MAGIC In the past, when examining a problem, you would have to manually keep track of the many models you created, as well as their associated parameters and metrics. This can quickly become tedious and take up valuable time, which is where MLflow comes in.
# MAGIC 
# MAGIC MLflow is pre-installed on the Databricks Runtime for ML.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC * Use MLflow to track experiments, log metrics, and compare runs

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.0 Imports

# COMMAND ----------

# MAGIC %run ../01_CONFIG/utils

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.0 Load the data

# COMMAND ----------

# Load the data and split between X and y
train_df = spark.sql("SELECT * FROM default.fish_cleaned_training").toPandas()

X = train_df.drop(model_config['TARGET_VARIABLE'], axis=1)
y = train_df[model_config['TARGET_VARIABLE']]

# Perform the train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=model_config['TEST_SIZE'], random_state = 42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.0 Train a regressor model without hyperparameter tuning
# MAGIC Let's take a step back and remember the usual way we train our models

# COMMAND ----------

# Create and instance
model = RandomForestRegressor()

# Fit the model
model.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1 Validate

# COMMAND ----------

# Perform predictions
y_pred = model.predict(X_test)

# Get an evaluation metric
mape = round(mean_absolute_percentage_error(y_test, y_pred), 3)
r2 = round(r2_score(y_test, y_pred), 3)

# COMMAND ----------

# Plot the scatter plot True versus Predicted
fig, axs = plt.subplots(figsize=(12, 8))

plt.scatter(y_test, y_pred)
plt.title(f"Predicted versus Ground truth\nR2 = {r2} | MAPE = {mape}")
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.2 Train the model with the optimal parameters

# COMMAND ----------

# Define some parameters
max_depth = 5
min_samples_leaf = 3
min_samples_split = 3
n_estimators = 200
random_state = 42

# COMMAND ----------

# Create and instance
model = RandomForestRegressor(
  max_depth = 5,
  min_samples_leaf = 3,
  min_samples_split = 3,
  n_estimators = 200,
  random_state = 42
)

# Fit the model
model.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.3 Validate

# COMMAND ----------

# Perform predictions
y_pred = model.predict(X_test)

# Get an evaluation metric
mape = round(mean_absolute_percentage_error(y_test, y_pred), 3)
r2 = round(r2_score(y_test, y_pred), 3)

# COMMAND ----------

# Plot the scatter plot True versus Predicted
# Plot the scatter plot True versus Predicted
fig, axs = plt.subplots(figsize=(12, 8))

plt.scatter(y_test, y_pred)
plt.title(f"Predicted versus Ground truth\nR2 = {r2} | MAPE = {mape}")
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.0 Using MLflow
# MAGIC <i18n value="9ab8c080-9012-4f38-8b01-3846c1531a80"/>
# MAGIC 
# MAGIC #### MLflow Tracking
# MAGIC 
# MAGIC MLflow Tracking is a logging API specific for machine learning and agnostic to libraries and environments that do the training.  It is organized around the concept of **runs**, which are executions of data science code.  Runs are aggregated into **experiments** where many runs can be a part of a given experiment and an MLflow server can host many experiments.
# MAGIC 
# MAGIC You can use <a href="https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_experiment" target="_blank">mlflow.set_experiment()</a> to set an experiment, but if you do not specify an experiment, it will automatically be scoped to this notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.1 Just run it simple

# COMMAND ----------

RUN_NAME = 'RandomForest'

# COMMAND ----------

# Run it with MLFlow
with mlflow.start_run(run_name = RUN_NAME):
  
  model = RandomForestRegressor(
  max_depth = 5,
  min_samples_leaf = 3,
  min_samples_split = 3,
  n_estimators = 200,
  random_state = 42
  )
  
  model.fit(X_train, y_train)
  
  # Perform predictions
  y_pred = model.predict(X_test)

  # Get an evaluation metric
  mape = round(mean_absolute_percentage_error(y_test, y_pred), 3)
  r2 = round(r2_score(y_test, y_pred), 3)
  
  # Plot the scatter plot True versus Predicted
  # Plot the scatter plot True versus Predicted
  fig, axs = plt.subplots(figsize=(12, 8))

  plt.scatter(y_test, y_pred)
  plt.title(f"Predicted versus Ground truth\nR2 = {r2} | MAPE = {mape}")
  plt.xlabel("True values")
  plt.ylabel("Predicted values")
  plt.show()

# COMMAND ----------

# MAGIC %md <i18n value="82786653-4926-4790-b867-c8ccb208b451"/>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC #### Track Runs
# MAGIC 
# MAGIC Each run can record the following information:<br><br>
# MAGIC 
# MAGIC - **Parameters:** Key-value pairs of input parameters such as the number of trees in a random forest model
# MAGIC - **Metrics:** Evaluation metrics such as RMSE or Area Under the ROC Curve
# MAGIC - **Artifacts:** Arbitrary output files in any format.  This can include images, pickled models, and data files
# MAGIC - **Source:** The code that originally ran the experiment
# MAGIC 
# MAGIC **NOTE**: For Spark models, MLflow can only log PipelineModels.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.2 Try to log some metrics and parameters

# COMMAND ----------

# Run it with MLFlow
with mlflow.start_run(run_name = RUN_NAME+'_with_logging'):
  
  model = RandomForestRegressor(
  max_depth = 5,
  min_samples_leaf = 3,
  min_samples_split = 3,
  n_estimators = 200,
  random_state = 42
  )
  
  model.fit(X_train, y_train)
  
  # Perform predictions
  y_pred = model.predict(X_test)

  # Get an evaluation metric
  mape = round(mean_absolute_percentage_error(y_test, y_pred), 3)
  r2 = round(r2_score(y_test, y_pred), 3)
  
  # Plot the scatter plot True versus Predicted
  # Plot the scatter plot True versus Predicted
  fig, axs = plt.subplots(figsize=(12, 8))

  plt.scatter(y_test, y_pred)
  plt.title(f"Predicted versus Ground truth\nR2 = {r2} | MAPE = {mape}")
  plt.xlabel("True values")
  plt.ylabel("Predicted values")
  #plt.show()
  
  
  # Log metrics
  mlflow.log_metric("MAPE", mape)
  mlflow.log_metric("R2", r2)
  
  # Log parameters
  mlflow.log_param("max_depth", max_depth)
  mlflow.log_param("n_estimators", n_estimators)
  
  # Log model
  mlflow.sklearn.log_model(model, "simple_randomforest")
  
  # Log the figure
  plt.savefig("r2_figure.png")
  mlflow.log_artifact("r2_figure.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.0 Query and compare past runs
# MAGIC 
# MAGIC You can query past runs programmatically in order to use this data back in Python.  The pathway to doing this is an **`MlflowClient`** object.

# COMMAND ----------

from mlflow...

# COMMAND ----------

# You can list the experiments using the search_experiments() method.
# It will return all the experiment available withing the Server
client = ...

# COMMAND ----------

# Query past runs within this notebook
runs = ...
runs.head()

# COMMAND ----------

# Query past runs with the Experiment ID
experiment_id = run...

# COMMAND ----------

runs = ...

# COMMAND ----------

# MAGIC %md
# MAGIC #### That's all for today!

# COMMAND ----------


