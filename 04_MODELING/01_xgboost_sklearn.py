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

fishDF = pd.read_csv("../Fish.csv")

# COMMAND ----------

fishDF.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.0 Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1 Pandas Get Dummies on the Species Column

# COMMAND ----------

species_dummy_df = pd.get_dummies(data=fishDF[["Species"]])

# COMMAND ----------

fishDFFeaturized = fishDF.join(species_dummy_df)
fishDFFeaturized = fishDFFeaturized.drop("Species", axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.0 Train Model

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.1 Train test Split

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(
    fishDFFeaturized.drop("Weight", axis=1),
    fishDFFeaturized["Weight"],
    test_size=0.2
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.2 Train the model

# COMMAND ----------

model = XGBRegressor()
    
model.fit(
    X_train,
    y_train
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.0 Evaluate the model

# COMMAND ----------

predictions = model.predict(X_test)

# COMMAND ----------

print(f"MAE: {mean_absolute_error(y_test, predictions)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions))}")
print(f"MSE: {mean_squared_error(y_test, predictions)}")

# COMMAND ----------



# COMMAND ----------

testingDF = X_test.copy()
testingDF['Weight'] = y_test
testingDF["Predictions"] = predictions
testingDF["Residual"] = testingDF["Weight"] - testingDF["Predictions"]

# COMMAND ----------

testingDF.head()

# COMMAND ----------

sns.scatterplot(
    data=testingDF,
    x='Weight',
    y='Predictions',
)

# COMMAND ----------

sns.histplot(
    data=testingDF,
    x='Residual',
    kde=True
)
plt.show()

# COMMAND ----------


