# Scoutium_Classification_of_Talent_Hunting_with_Machine_Learning

## Business Problem

The goal of this project is to predict whether football players are classified as "average" or "highlighted" based on the ratings given by scouts to players' attributes.

## Dataset Story

The dataset consists of information collected by scouts during football matches, including ratings of players' attributes and potential labels assigned by scouts.

- **attributes**: Contains ratings given by scouts to the attributes of each player in a match (independent variables).
- **potential_labels**: Contains potential labels assigned by scouts for each player in a match (target variable).
- Total Variables: 9
- Total Observations: 10,730
- Dataset Size: 0.65 MB

## Variables

- **task_response_id**: Set of evaluations by a scout for all players in a team in a match.
- **match_id**: ID of the relevant match.
- **evaluator_id**: ID of the evaluator (scout).
- **player_id**: ID of the player.
- **position_id**: ID of the player's position in that match.
    - 1- Goalkeeper
    - 2- Center-back
    - 3- Right-back
    - 4- Left-back
    - 5- Defensive midfielder
    - 6- Central midfielder
    - 7- Right winger
    - 8- Left winger
    - 9- Attacking midfielder
    - 10- Forward
- **analysis_id**: Set of attribute evaluations by a scout for a player in a match.
- **attribute_id**: ID of each attribute for which players are evaluated.
- **attribute_value**: Rating given by a scout to a player's attribute.
- **potential_label**: Label assigned by a scout for a player in a match (target variable).

## Data Preparation

1. Read two CSV files.
2. Merge the CSV files using specific columns ("task_response_id", 'match_id', 'evaluator_id', "player_id").
3. Remove the "Goalkeeper" class from the "position_id" column.
4. Remove the "below_average" class from the "potential_label" column.
5. Create a pivot table from the modified dataset.
6. Reset the index and convert the "attribute_id" columns to strings.

## Exploratory Data Analysis (EDA)

1. Overview:
   - Display dataset shape, data types, head, tail, and missing values.
   - Show quantiles for numerical variables.
2. Analyze categorical and numerical variables.
3. Analysis of Categorical Variables:
   - Display counts and ratios for "position_id" and "potential_label."
4. Analysis of Numerical Variables:
   - Display statistics and histograms for numerical variables.
5. Analyze the relationship between numerical variables and the target variable.
6. Check the correlation between numerical variables using a heatmap.

## Feature Extraction

- Create new features such as minimum, maximum, sum, mean, median, mentality, counts, and count ratio based on existing data.

## Label Encoding

- Apply label encoding for "potential_label" and "mentality" categories ("average" and "highlighted").

## Standard Scaling

- Apply StandardScaler to scale all numerical columns.

## Model Development

- Train various machine learning models (Logistic Regression, K-Nearest Neighbors, Random Forest, Gradient Boosting, XGBoost) and evaluate their performance using cross-validation.

## Hyperparameter Optimization

- Perform hyperparameter optimization for the selected model using GridSearchCV.

## Plot Feature Importances

- Visualize feature importances using bar plots.

## Conclusion

- The final model is trained and evaluated for potential label prediction with minimal error.

---

This documentation provides an overview of the data preparation, exploratory data analysis, feature extraction, model development, hyperparameter optimization, and feature importances in the Classification of Talent Hunting with Machine Learning project.
