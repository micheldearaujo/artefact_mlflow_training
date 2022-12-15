# Databricks notebook source
# MAGIC %md
# MAGIC ## Model Registry
# MAGIC 
# MAGIC The MLflow Model Registry component is a centralized model store, set of APIs, and UI, to collaboratively manage the full lifecycle of an MLflow Model. It provides model lineage (which MLflow experiment and run produced the model), model versioning, stage transitions (for example from staging to production), and annotations.
# MAGIC 
# MAGIC **Objective**: This notebook's objectives is to learn how to
# MAGIC 
# MAGIC - Query experiment runs;
# MAGIC - Register a model to MLflow Models;
# MAGIC - Transit model to Staging/Production stages
# MAGIC - Perform model selection between multiple runs

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.0 Imports

# COMMAND ----------

# MAGIC %run ../01_CONFIG/utils

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.0 Load the past experiment runs

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.1 Define the paths

# COMMAND ----------

# Write down your username
user = ''
# Adapt your workspace here - Repos or Workspace + email
# Get the experiment IDs
experiments_notebook = mlflow.get_experiment_by_name(f'/Repos/{user}/artefact_mlflow_training/04_MODELING/02_Intro_to_Hyperopt')
experiments_ids = experiments_notebook.experiment_id

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2 Load the runs

# COMMAND ----------

from mlflow import MlflowClient
from mlflow.entities import ViewType

# Create a Client
client = MlflowClient()

# Use the search_runs method to search the past runs
runs = mlflow.search_runs(
  [experiments_ids],
  order_by=[f"metrics.{model_config['COMPARISON_METRIC']} ASC"]
)

# Filter the runs to only include the finished ones and the runs of the day
runs = runs[(runs['status'] == 'FINISHED')&\
             (runs['end_time'].dt.strftime('%Y-%m-%d') == dt.datetime.today().strftime('%Y-%m-%d'))]

# COMMAND ----------

runs.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.3 Get the best run
# MAGIC 
# MAGIC To get the best run is simply as selecting the first row (because we sorted ascending)

# COMMAND ----------

# Get the best run
best_run = runs.iloc[0]
# Get the best run ID
best_run_id = best_run.run_id
print(f"Best run ID: {best_run_id}")
# Get the best run name - Cool if you are running multiple models
best_model_name = best_run['tags.mlflow.runName']
print(f"Best Model: {best_model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.4 Load this Model
# MAGIC 
# MAGIC We can load the model using the logging info: The Run ID and the Model Name. When using MLflow, we do not need to manually save Pickle files.

# COMMAND ----------

model = mlflow.sklearn.load_model(model_uri = f"runs:/{best_run_id}/{best_model_name}")
model

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.5 Register the model
# MAGIC 
# MAGIC We use the `register_model` method to register a model. For that, we need the URI (the same that we used to load the model), and a name for that given model.
# MAGIC 
# MAGIC The model name here can be any name you want.

# COMMAND ----------

# Register model
result = mlflow.register_model(
    model_uri = f"runs:/{best_run_id}/{best_model_name}",
    name = model_config['REGISTER_MODEL_NAME']
)

# COMMAND ----------

# MAGIC %md
# MAGIC Take a look at the Databricks Models UI!!

# COMMAND ----------

# MAGIC %md
# MAGIC ##### We can also update the model description by code

# COMMAND ----------

# Add a description to the model
client.update_model_version(
    name=model_config['REGISTER_MODEL_NAME'],
    version=result.version,
    description="ATF Meetup Model Info -> Model Type = {}".format(model)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.0 Compare the new best run with the last Production/Staging model

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.1 Set this model to the Staging
# MAGIC 
# MAGIC We use the `client.transition_model_version_stage` to change the model's stage.
# MAGIC 
# MAGIC We need to specify the model name (the Generic one), the model version that we want to setup and the stage name, that can be `Staging` or `Production`

# COMMAND ----------

client.transition_model_version_stage(
  name=model_config['REGISTER_MODEL_NAME'],
  version=result.version,
  stage='Staging',
  )

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.2 Load the Staging model
# MAGIC 
# MAGIC We can search for only the models that are in a specific stage. For example, we will load the Staging model.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### The first step is to create a list with all the model versions that we have so far

# COMMAND ----------

models_versions = []

# COMMAND ----------

for mv in client.search_model_versions("name='{}'".format(model_config['REGISTER_MODEL_NAME'])):
  models_versions.append(dict(mv))

# COMMAND ----------

models_versions

# COMMAND ----------

# Filter the mmodels that are on Staging
current_model = [model for model in models_versions if model['current_stage'] == 'Staging'][0]

# COMMAND ----------

current_model

# COMMAND ----------

# Get the current Staging model
current_model = [x for x in models_versions if x['current_stage'] == 'Staging'][0]

# Extract the current staging model MAPE
current_model_mape = mlflow.get_run(current_model['run_id']).data.metrics[model_config['COMPARISON_METRIC']]

# Get the new model MAPE
candidate_model_mape = mlflow.get_run(result.run_id).data.metrics[model_config['COMPARISON_METRIC']]

# COMMAND ----------

# MAGIC %md
# MAGIC If the newest run beats the oldest, then transition the last Staging version to Archived

# COMMAND ----------

if candidate_model_mape < current_model_mape:
    print(f"Candidate model has a better {model_config['COMPARISON_METRIC']} than the active model. Switching models...")
    
    client.transition_model_version_stage(
        name=model_config['REGISTER_MODEL_NAME'],
        version=result.version,
        stage='Staging',
    )

    client.transition_model_version_stage(
        name=model_config['REGISTER_MODEL_NAME'],
        version=current_model['version'],
        stage='Archived',
    )
else:
    print(f"Active model has a better {model_config['COMPARISON_METRIC']} than the candidate model. No changes to be applied.")
    
print(f"Candidate: {model_config['COMPARISON_METRIC']} = {candidate_model_mape}\nCurrent: = {current_model_mape}")

# COMMAND ----------


