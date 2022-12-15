# Databricks notebook source
# MAGIC %md
# MAGIC ## Hyperopt: Distributed Asynchronous Hyper-parameter Optimization
# MAGIC 
# MAGIC Databricks Runtime ML includes Hyperopt, a Python library that facilitates distributed hyperparameter tuning and model selection. With Hyperopt, you can scan a set of Python models while varying algorithms and hyperparameters across spaces that you define. Hyperopt works with both distributed ML algorithms such as Apache Spark MLlib and Horovod, as well as with single-machine ML models such as scikit-learn and TensorFlow.
# MAGIC 
# MAGIC **Objective**: This notebook's objective is train and optimise a Random Forest regression model
# MAGIC 
# MAGIC The basic steps when using Hyperopt are:
# MAGIC 
# MAGIC - Define an objective function to minimize. Typically this is the training or validation loss.
# MAGIC - Define the hyperparameter search space. Hyperopt provides a conditional search space, which lets you compare different ML algorithms in the same run.
# MAGIC - Specify the search algorithm. Hyperopt uses stochastic tuning algorithms that perform a more efficient search of hyperparameter space than a deterministic grid search.
# MAGIC - Run the Hyperopt function fmin(). fmin() takes the items you defined in the previous steps and identifies the set of hyperparameters that minimizes the objective function.
# MAGIC 
# MAGIC #### Resources:
# MAGIC 
# MAGIC http://hyperopt.github.io/hyperopt/
# MAGIC 
# MAGIC https://learn.microsoft.com/en-us/azure/databricks/machine-learning/automl-hyperparam-tuning/hyperopt-concepts
# MAGIC 
# MAGIC https://learn.microsoft.com/en-us/azure/databricks/machine-learning/automl-hyperparam-tuning/#hyperopt-overview
# MAGIC 
# MAGIC https://learn.microsoft.com/en-us/azure/databricks/machine-learning/automl-hyperparam-tuning/hyperopt-best-practices

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.0 Imports

# COMMAND ----------

# MAGIC %run ../01_CONFIG/utils

# COMMAND ----------

# Give a name for this run
RUN_NAME = 'RandomForest_Hyperopt'

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.0 Load the data

# COMMAND ----------

train_df = spark.sql("SELECT * FROM default.fish_cleaned_training").toPandas()
X = train_df.drop(model_config['TARGET_VARIABLE'], axis=1)
y = train_df[model_config['TARGET_VARIABLE']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=model_config['TEST_SIZE'], random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.0 Build the hyperparameter optimisation

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1 Define the Grid

# COMMAND ----------

# Create a dictionary with the parameter name and the values
# Use hp.choice for a list of integers or hp.uniform for a range of floats
randomforest_hyperparameter_config = {
    'bootstrap': hp.choice('bootstrap', [True, False]),
    'max_depth': hp.choice('max_depth', [10, 20, 30, 90, 100, None]),
    'max_features': hp.choice('max_features', ['auto', 'sqrt']),
    'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 5]),
    'min_samples_split': hp.choice('min_samples_split', [2, 5, 20]),
    'n_estimators': hp.choice('n_estimators', [200, 400, 800, 1000])
}

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.2 Define the objective function
# MAGIC Now we need to create a function that will run multiple times in order to find the best parameters. This function minimizes some metric.

# COMMAND ----------

def objective(search_space):
  """
  Train and test an random forest with a set of parameters
  from the Grid.
  
  Returns: The resulting metric for an iteration.
  """
  
  # Create the model instance, with parameters from the grid
  model = RandomForestRegressor(
      random_state = random_forest_fixed_model_config['RANDOM_STATE'], # Some fixed model parameters
      **search_space ## Specify the search_space here
  )
  
 # Train the model
  model.fit(
      X_train,
      y_train
  )
  
  # Perform predictions and extract the desired optimisation metric
  y_pred = model.predict(X_test)
  mse = mean_squared_error(y_test, y_pred)
  
  # Return the metrics
  return {'loss': mse, 'status': STATUS_OK}

# COMMAND ----------

# Get the search_space
search_space = randomforest_hyperparameter_config

# Define which algorithm to optimise. In this case, we will use the # Tree of Parzen Estimators, that is bayesian
algorithm = tpe.suggest 

# The Trials Object. SparkTrials for single-node algorithms (sklearn) or Trials for distributed libraries (SparkML)
spark_trials = SparkTrials(parallelism=model_config['PARALELISM']) 

# COMMAND ----------

# Now perform the otimisation, using the fmin function

with mlflow.start_run(run_name=RUN_NAME):
  
    best_params = fmin(
        fn=objective, # Objective function
        space=search_space, # Search space
        algo=algorithm, # Algorithm
        max_evals=50, #model_config['MAX_EVALS'], # Max number of iterations
        trials=spark_trials # Spark Trails
    )
    
# This function will return the best parameters

# COMMAND ----------

best_params

# COMMAND ----------

# Get the best parameters
rf_best_param_names = space_eval(search_space, best_params)
rf_best_param_names

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.2 Train the model with the optimal parameters

# COMMAND ----------

with mlflow.start_run(run_name = RUN_NAME) as run:
    # First configure the fixed parameters, such as random_state
    random_state = random_forest_fixed_model_config['RANDOM_STATE']
    
    # Getting the best parameters configuration
    bootstrap = rf_best_param_names['bootstrap']
    max_depth = rf_best_param_names['max_depth']
    max_features = rf_best_param_names['max_features']
    min_samples_leaf = rf_best_param_names['min_samples_leaf']
    min_samples_split = rf_best_param_names['min_samples_split']
    n_estimators = rf_best_param_names['n_estimators']

    # Create the model instance if the selected parameters
    model = RandomForestRegressor(
        bootstrap = bootstrap,
        max_depth = max_depth,
        max_features = max_features,
        min_samples_leaf = min_samples_leaf,
        min_samples_split = min_samples_split,
        n_estimators = n_estimators,
        random_state = random_state,
    )

    # Training the model
    model_fit = model.fit(
        X=X_train,
        y=y_train
    )

    ### Perform Predictions
    # Use the model to make predictions on the test dataset.
    predictions = model_fit.predict(X_test)

    ### Log the metrics

    mlflow.log_param("bootstrap", bootstrap)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("max_features", max_features)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("min_samples_leaf", min_samples_leaf)
    mlflow.log_param("min_samples_split", min_samples_split)
    mlflow.log_param("n_estimators", n_estimators)

    # Define a metric to use to evaluate the model.

    # RMSE
    rmse = round(np.sqrt(mean_squared_error(y_test, predictions)), 2)
    # R2
    r2 = round(r2_score(y_test, predictions), 2)
    # R2 adjusted
    p = X_test.shape[1]
    n = X_test.shape[0]
    adjust_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    # MAPE
    mape = round(mean_absolute_percentage_error(y_test, predictions), 3)


    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("R2_Adj", adjust_r2)
    mlflow.log_metric("MAPE", mape)

    mlflow.log_metric('Dataset_Size', X_train.shape[0])
    mlflow.log_metric('Number_of_variables', X_train.shape[1])

    fig, axs = plt.subplots(figsize=(12, 8))
    axs.scatter(x=y_test, y=predictions)
    axs.set_title(f"Random Forest Predicted versus ground truth\n R2 = {r2} | RMSE = {rmse} | MAPE = {mape}")
    axs.set_xlabel(f"True {model_config['TARGET_VARIABLE']}")
    axs.set_ylabel(f"Predicted {model_config['TARGET_VARIABLE']}")
    plt.savefig("scatter_plot_rf.png")
    fig.show()

    mlflow.log_artifact("scatter_plot_rf.png")

    mlflow.sklearn.log_model(model_fit, RUN_NAME)

    np.savetxt('predictions_rf.csv', predictions, delimiter=',')

    # Log the saved table as an artifact
    mlflow.log_artifact("predictions_rf.csv")

    # Convert the residuals to a pandas dataframe to take advantage of graphics  
    predictions_df = pd.DataFrame(data = predictions - y_test)

    plt.figure()
    plt.plot(predictions_df)
    plt.xlabel("Observation")
    plt.ylabel("Residual")
    plt.title("Residuals")

    plt.savefig("residuals_plot_rf.png")
    mlflow.log_artifact("residuals_plot_rf.png")

# COMMAND ----------

model

# COMMAND ----------


