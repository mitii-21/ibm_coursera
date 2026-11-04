# SpaceX Falcon 9 First-Stage Landing Prediction — Results Draft

## Objective
Predict whether a Falcon 9 first stage lands successfully using the IBM course-prepared dataset.

## Dataset and preparation
The analysis used 90 records and 21 encoded predictor columns. The target variable was `Class`. Rows without a target value were excluded; categorical predictors were one-hot encoded and missing numeric values were imputed with the median.

## Model evaluation
Models were tuned using GridSearchCV with 3-fold cross-validation and evaluated on a held-out 20% test split (random_state=42).

| Model               |   Best CV F1 |   Test Accuracy |   Precision |   Recall |       F1 | Best Parameters                                                       |
|:--------------------|-------------:|----------------:|------------:|---------:|---------:|:----------------------------------------------------------------------|
| Decision Tree       |     0.881631 |        0.888889 |    0.916667 | 0.916667 | 0.916667 | {'model__max_depth': 2, 'model__min_samples_split': 2}                |
| Logistic Regression |     0.863815 |        0.833333 |    0.8      | 1        | 0.888889 | {'model__C': 0.1}                                                     |
| KNN                 |     0.888815 |        0.833333 |    0.8      | 1        | 0.888889 | {'model__n_neighbors': 7, 'model__weights': 'distance'}               |
| SVM                 |     0.888815 |        0.777778 |    0.75     | 1        | 0.857143 | {'model__C': 0.1, 'model__gamma': 'scale', 'model__kernel': 'linear'} |

Best model by held-out F1 score: **Decision Tree**.

## Limitations
This is a course-dataset demonstration. Results depend on the prepared dataset, feature availability, random split and chosen metrics. The model is not production-ready and should not be interpreted causally. Inspect the confusion matrix and class balance before drawing conclusions.

## Figures
- `01_landing_class_distribution.png`
- `02_payload_mass_by_outcome.png` (if payload mass is present)
- `04_correlation_heatmap.png` (if sufficient numeric columns are present)
- `05_best_model_confusion_matrix.png`
- `launch_sites_map.html` and `interactive_dashboard.html`