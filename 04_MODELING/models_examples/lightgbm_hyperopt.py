# Databricks notebook source
# MAGIC %md
# MAGIC ## LightGBM Hyperparameter Optimisation
# MAGIC 
# MAGIC **Objective**: This notebook's objective is train and optimise a Light Gradient Boosting Machine regression model

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.0 Imports

# COMMAND ----------

# MAGIC %run ../01_CONFIG/utils

# COMMAND ----------

RUN_NAME = 'LightGBM_Hyperopt'

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.0 Build the hyperparameter optimisation

# COMMAND ----------

def objective(search_space):
    
    model = LGBMRegressor(
        learning_rate= lgbm_fixed_model_config['LEARNING_RATE'],
        min_data = lgbm_fixed_model_config['MIN_DATA'],
        subsample = lgbm_fixed_model_config['SUBSAMPLE'],
        num_iterations = lgbm_fixed_model_config['NUM_ITERATIONS'],
        seed = lgbm_fixed_model_config['SEED'],
        **search_space
    )
    model.fit(
        X_train,
        y_train,
        early_stopping_rounds = 200,
        eval_metric = ['rmse'],
        eval_set=[[X_train, y_train],[X_test, y_test]]
    )
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    return {'loss': mse, 'status': STATUS_OK}

# COMMAND ----------

search_space = lightgbm_hyperparameter_config

algorithm = tpe.suggest

spark_trials = SparkTrials(parallelism=model_config['PARALELISM'])

# COMMAND ----------

with mlflow.start_run(run_name=RUN_NAME):
    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=algorithm,
        max_evals=model_config['MAX_EVALS'],
        trials=spark_trials
    )

# COMMAND ----------

lightgbm_best_param_names = space_eval(search_space, best_params)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.2 Train the model with the optimal parameters

# COMMAND ----------

with mlflow.start_run(run_name = RUN_NAME) as run:
    
    learning_rate= lgbm_fixed_model_config['LEARNING_RATE'],
    min_data = lgbm_fixed_model_config['MIN_DATA'],
    subsample = lgbm_fixed_model_config['SUBSAMPLE'],
    num_iterations = lgbm_fixed_model_config['NUM_ITERATIONS'],
    seed = lgbm_fixed_model_config['SEED'],

    # Getting the best parameters configuration
    try:
        boosting_type= lightgbm_best_param_names['boosting_type'],
        max_depth = lightgbm_best_param_names['max_depth'],
        colsample_bytree = lightgbm_best_param_names['colsample_bytree'],
        n_estimators = lightgbm_best_param_names['n_estimators'],
        num_leaves = lightgbm_best_param_names['num_leaves'],
        reg_lambda = lightgbm_best_param_names['reg_lambda'],
        scale_pos_weight = lightgbm_best_param_names['scale_pos_weight']
        print("went here")
        
    # If something goes wrong, select the pre-selected parameters in the config file
    except:
        boosting_type= lgbm_model_config['boosting_type'],
        max_depth = lgbm_model_config['max_depth'],
        colsample_bytree = lgbm_model_config['colsample_bytree'],
        n_estimators = lgbm_model_config['n_estimators'],
        num_leaves = lgbm_model_config['num_leaves'],
        reg_lambda = lgbm_model_config['reg_lambda'],
        scale_pos_weight = lgbm_model_config['scale_pos_weight']
        print("otherwise")

    # Create the model instance if the selected parameters
    n_estimators = n_estimators[0]
    model = LGBMRegressor(
        boosting_type = boosting_type,
        learning_rate = learning_rate,
        max_depth = max_depth,
        reg_lambda = reg_lambda,
        n_estimators = n_estimators,
        num_leaves = num_leaves,
        scale_pos_weight = scale_pos_weight,
        colsample_bytree = colsample_bytree,
        seed = seed,
        subsample = subsample,
        #num_iterations = num_iterations,
    )

    # Training the model
    model_fit = model.fit(
        X=X_train,
        y=y_train,
        callbacks = [lgb.early_stopping(100)],
        eval_metric= ['mape'],
        eval_set =[(X_train, y_train), (X_test, y_test)]
    )

    ### Perform Predictions
    # Use the model to make predictions on the test dataset.
    predictions = model_fit.predict(X_test)

    ### Log the metrics
    
    mlflow.log_param("boosting_type", boosting_type)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_data", min_data)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("reg_lambda", reg_lambda)
    mlflow.log_param("num_leaves", num_leaves)
    mlflow.log_param("scale_pos_weight", scale_pos_weight)
    mlflow.log_param("colsample_bytree", colsample_bytree)
    mlflow.log_param("num_iterations", num_iterations)

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
    axs.set_title(f"XGBoost Predicted versus ground truth\n R2 = {r2} | RMSE = {rmse} | MAPE = {mape}")
    axs.set_xlabel(f"True {model_config['TARGET_VARIABLE']}")
    axs.set_ylabel(f"Predicted {model_config['TARGET_VARIABLE']}")
    plt.savefig("scatter_plot_lightgbm.png")
    fig.show()

    mlflow.log_artifact("scatter_plot_lightgbm.png")

    mlflow.sklearn.log_model(model_fit, RUN_NAME)

    np.savetxt('predictions_lightgbm.csv', predictions, delimiter=',')

    # Log the saved table as an artifact
    mlflow.log_artifact("predictions_lightgbm.csv")

    # Convert the residuals to a pandas dataframe to take advantage of graphics  
    predictions_df = pd.DataFrame(data = predictions - y_test)

    plt.figure()
    plt.plot(predictions_df)
    plt.xlabel("Observation")
    plt.ylabel("Residual")
    plt.title("Residuals")

    plt.savefig("residuals_plot_lightgbm.png")
    mlflow.log_artifact("residuals_plot_lightgbm.png")

# COMMAND ----------


