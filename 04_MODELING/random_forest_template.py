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

# Load the data 
train_df = spark.sql("SELECT * FROM default.fish_cleaned_training").toPandas()

# COMMAND ----------

# split between X and y
X = train_df.drop(model_config['TARGET_VARIABLE'], axis=1)
y = train_df[model_config['TARGET_VARIABLE']]

# Perform the train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=model_config['TEST_SIZE'], random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.0 Train a regressor model without hyperparameter tuning

# COMMAND ----------

# Create and instance
model = RandomForestRegressor()

# Fit the model
model.fit(
X_train,
y_train
)

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
    max_depth = max_depth,
    min_samples_leaf = min_samples_leaf,
    min_samples_split = min_samples_split,
    n_estimators = n_estimators,
    random_state = random_state,
)

# Fit the model
model.fit(
X_train,
y_train
)


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

fig, axs = plt.subplots(figsize=(12, 8))

plt.scatter(y_test, y_pred)
plt.title(f"Predicted versus Ground truth - With some optimal parameters\nR2 = {r2} | MAPE = {mape}")
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.0 Using MLflow Tracking
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

with mlflow.start_run(run_name = RUN_NAME) as run:
  
    # Create and instance
    model = RandomForestRegressor(
        max_depth = max_depth,
        min_samples_leaf = min_samples_leaf,
        min_samples_split = min_samples_split,
        n_estimators = n_estimators,
        random_state = random_state,
    )

    # Fit the model
    model.fit(
    X_train,
    y_train
    )
    
    ## Validate the model
    # Perform predictions
    y_pred = model.predict(X_test)

    # Get an evaluation metric
    mape = round(mean_absolute_percentage_error(y_test, y_pred), 3)
    r2 = round(r2_score(y_test, y_pred), 3)
    

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

with mlflow.start_run(run_name = RUN_NAME+'_with_log') as run:
    # Create and instance
    model = RandomForestRegressor(
        max_depth = max_depth,
        min_samples_leaf = min_samples_leaf,
        min_samples_split = min_samples_split,
        n_estimators = n_estimators,
        random_state = random_state,
    )

    # Fit the model
    model.fit(
    X_train,
    y_train
    )
    
    ## Validate the model
    # Perform predictions
    y_pred = model.predict(X_test)

    # Get an evaluation metric
    mape = round(mean_absolute_percentage_error(y_test, y_pred),3 )
    r2 = round(r2_score(y_test, y_pred), 3)
    
    # Plot the regression results
    fig, axs = plt.subplots(figsize=(12, 8))

    plt.scatter(y_test, y_pred)
    plt.title(f"Predicted versus Ground truth\nR2 = {r2} | MAPE = {mape}")
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    
    # Save the figure to log it later
    plt.savefig("r2_figure.png")
    plt.show()
    
    # Log the metrics
    mlflow.log_metric("MAPE", mape)
    mlflow.log_metric("R2", r2)
    
    # Log the parameters
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_samples_leaf", min_samples_leaf)
    mlflow.log_param("min_samples_split", min_samples_split)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)
    
    # Log the model
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    # Now log the figure
    mlflow.log_artifact('r2_figure.png')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.0 Query past runs

# COMMAND ----------

from mlflow.tracking import MlflowClient

# COMMAND ----------

# Create a Client
client = MlflowClient()

# COMMAND ----------

# You can list the experiments using the list_experiments() method.
# It will return all the experiment available withing the Server
client.search_experiments()

# COMMAND ----------

# To query past runs within this experiment (this notebook)
runs = mlflow.search_runs()

# COMMAND ----------

runs.head()

# COMMAND ----------

# The search_runs function has tons of parameters. To search the experiments in another notebook, you have to get the Experiment ID.
# Later we will see how to get experiments from another notebook
# We can use the last run to get the experiment id
experiment_id = run.info.experiment_id

# COMMAND ----------

runs = mlflow.search_runs(
  experiment_ids =[experiment_id],
 order_by = ["metrics.MAPE ASC"] # We can order by some column
 )

# COMMAND ----------

runs

# COMMAND ----------

# MAGIC %md
# MAGIC #### That's all for today!!

# COMMAND ----------


