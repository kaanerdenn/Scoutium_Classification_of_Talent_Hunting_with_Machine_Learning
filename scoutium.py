################################################################
# CLASSIFICATION OF TALENT HUNTING WITH MACHINE LEARNING
################################################################

################################################################
# Business Problem:
################################################################
# Predicting whether football players are classified as "average" or "highlighted" based on the ratings given by scouts to players' attributes.

################################################################
# Dataset Story:
################################################################

# The dataset consists of information collected by scouts during football matches, including ratings of players' attributes and potential labels assigned by scouts.

# attributes: Contains ratings given by scouts to the attributes of each player in a match. (independent variables)
# potential_labels: Contains potential labels assigned by scouts for each player in a match. (target variable)
# 9 Variables, 10730 Observations, 0.65 MB

################################################################
# Variables:
################################################################

# task_response_id: Set of evaluations by a scout for all players in a team in a match.

# match_id: ID of the relevant match.

# evaluator_id: ID of the evaluator (scout).

# player_id: ID of the player.

# position_id: ID of the player's position in that match.

# 1- Goalkeeper
# 2- Center-back
# 3- Right-back
# 4- Left-back
# 5- Defensive midfielder
# 6- Central midfielder
# 7- Right winger
# 8- Left winger
# 9- Attacking midfielder
# 10- Forward

# analysis_id: Set of attribute evaluations by a scout for a player in a match.

# attribute_id: ID of each attribute for which players are evaluated.

# attribute_value: Rating given by a scout to a player's attribute.

# potential_label: Label assigned by a scout for a player in a match (target variable).

import pandas as pd
import numpy as np
from sklearn.model_selection import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_predict
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

################################################################
# Task 1: Data Preparation
################################################################

################################################################
# Step 1: Read two CSV files.
################################################################

df = pd.read_csv("Scoutium-220805-075951/scoutium_attributes.csv", sep=";")

df2 = pd.read_csv("Scoutium-220805-075951/scoutium_potential_labels.csv", sep=";")

################################################################
# Step 2: Merge the CSV files using specific columns ("task_response_id", 'match_id', 'evaluator_id', "player_id").
################################################################

dff = pd.merge(df, df2, how='left', on=["task_response_id", 'match_id', 'evaluator_id', "player_id"])

################################################################
# Step 3: Remove the "Goalkeeper" class from the "position_id" column.
################################################################

dff = dff[dff["position_id"] != 1]

################################################################
# Step 4: Remove the "below_average" class from the "potential_label" column. (below_average class constitutes only 1% of the dataset)
################################################################

dff = dff[dff["potential_label"] != "below_average"]

################################################################
# Step 5: Create a pivot table from the modified dataset.
################################################################

pt = pd.pivot_table(dff, values="attribute_value", columns="attribute_id", index=["player_id","position_id","potential_label"])

################################################################
# Step 6: Reset the index and convert the "attribute_id" columns to strings.
################################################################

pt = pt.reset_index(drop=False)
pt.columns = pt.columns.map(str)

################################################################
# Task 3: Save numerical columns in a list called "num_cols."
################################################################

num_cols = pt.columns[3:]

##################################
# TASK 4: EXPLORATORY DATA ANALYSIS (EDA)
##################################

##################################
# Step 1: OVERVIEW
##################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(pt)

##################################
# Step 2: Analyze categorical and numerical variables.
##################################

##################################
# ANALYSIS OF CATEGORICAL VARIABLES
##################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in ["position_id","potential_label"]:
    cat_summary(pt, col)

##################################
# ANALYSIS OF NUMERICAL VARIABLES
##################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(pt, col, plot=True)

##################################
# Step 3: Analyze the relationship between numerical variables and the target variable.
##################################

##################################
# ANALYSIS OF NUMERICAL VARIABLES BY TARGET
##################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(pt, "potential_label", col)

##################################
# Step 4: Check the correlation between numerical variables.
##################################

pt[num_cols].corr()

# Correlation Matrix
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(pt[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

##################################
# Task 5: Feature Extraction
##################################

pt["min"] = pt[num_cols].min(axis=1)
pt["max"] = pt[num_cols].max(axis=1)
pt["sum"] = pt[num_cols].sum(axis=1)
pt["mean"] = pt[num_cols].mean(axis=1)
pt["median"] = pt[num_cols].median(axis=1)

pt["mentality"] = pt["position_id"].apply(lambda x: "defender" if (x == 2) | (x == 5) | (x == 3) | (x == 4) else "attacker")

flagCols = [col for col in pt.columns if "_FLAG" in col]

pt["counts"] = pt[flagCols].sum(axis=1)

pt["countRatio"] = pt["counts"] / len(flagCols)

pt.head()

################################################################
# Task 6: Label Encoding for "potential_label" and "mentality" categories ("average" and "highlighted").
################################################################

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

labelEncoderCols = ["potential_label", "mentality"]

for col in labelEncoderCols:
    pt = label_encoder(pt, col)

################################################################
# Task 7: Apply StandardScaler to scale all numerical columns.
################################################################

lst = ["counts", "countRatio", "min", "max", "sum", "mean", "median"]
num_cols = list(num_cols)

for i in lst:
    num_cols.append(i)

scaler = StandardScaler()
pt[num_cols] = scaler.fit_transform(pt[num_cols])

pt.head()

################################################################
# Task 8: Develop a Machine Learning Model to predict potential labels with minimal error.
################################################################

y = pt["potential_label"]
X = pt.drop(["potential_label", "player_id"], axis=1)

models = [('LR', LogisticRegression()),
          ('KNN', KNeighborsClassifier()),
          ('RF', RandomForestClassifier()),
          ('GBM', GradientBoostingClassifier()),
          ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))]

for name, model in models:
    print(name)
    for score in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
        cvs = cross_val_score(model, X, y, scoring=score, cv=10).mean()
        print(score + " score:" + str(cvs))

################################################################
# Task 9: Hyperparameter Optimization
################################################################

lgbm_model = LGBMClassifier(random_state=46)

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500],
               "colsample_bytree": [0.5, 0.7, 1]
               }

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X, y)

final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))

################################################################
# Task 10: Plot Feature Importances
################################################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

model = LGBMClassifier()
model.fit(X, y)

plot_importance(model, X)
