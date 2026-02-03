# Engage2Value: From Clicks to Conversions

> **Author**: Ankita Dey, 23f2001106                                       
> **Course**: Machine Learning Practice, BS in Data Science and Applications, IIT Madras

## Overview

Predict a customer’s purchase value using anonymized multi-session digital interaction data.  
Helps estimate **purchase potential** to support data-driven marketing and engagement strategies.

Dataset: 
[Kaggle Competition](https://www.kaggle.com/competitions/engage-2-value-from-clicks-to-conversions)

**Final Model Performance on Kaggle Test Set:**    
The final model achieved a score of **0.71**, demonstrating strong generalization to unseen data.

## Workflow and Methodology

<details>
<summary>Click to expand</summary>

1. **Exploratory Data Analysis (EDA)**
   - Understand data distribution, missing values, and patterns.

2. **Data Preprocessing**
   - Handle missing values
   - Encode categorical variables
   - Scale/normalize features

3. **Feature Engineering**
   - Create new features from existing ones
   - Generate interaction terms

4. **Polynomial Features**
   - Add degree-2 interaction terms for regression

5. **Feature Selection**
   - Use SelectKBest with mutual information to retain top predictors

6. **Model Training**
   - Train base models for classification and regression
   - Evaluate using F1, ROC AUC, MAE, RMSE, R²

7. **Ensemble Learning**
   - Combine top models via blending or stacking

8. **Prediction Strategy**
   - Soft prediction: probability × regression output
   - Direct prediction: regression output for purchasers

9. **Hyperparameter Tuning**
   - Optimize model parameters for better performance

10. **Final Predictions**
    - Generate purchase value predictions on test data

</details>


## Model Training and Evaluation

**Strategy:** Two-stage modeling — Classification followed by Regression  
**Metrics:** F1 score (classification) and R² (regression)

* **Classification:**  
  A lean set of top features provided high precision and recall, reliably identifying potential purchasers.

* **Regression:**  
  Engineered features and polynomial interactions improved the model’s ability to explain variance in purchase values.

* **Ensemble Learning:**  
  Blending top models balanced accuracy and robustness, capturing both local patterns (from KNN) and broader trends (from tree-based models).

* **Prediction Strategy:**  
  Direct regression predictions were chosen over soft predictions because the classifier was strong, maximizing explained variance.



## Hyperparameter Tuning

* Explored multiple methods: grid search, random search, and Bayesian optimization.
* Tuning focused on improving accuracy, reducing errors, and optimizing generalization.
* Smart hyperparameter selection was crucial for high-dimensional, interaction-heavy features.


## Conclusion


* Feature engineering and polynomial interactions significantly improved predictive performance.
* Feature selection removed noise, enhancing model generalization.
* KNN and ensemble blending outperformed traditional gradient boosting models in this interaction-rich feature space.
* Direct regression predictions provided the best estimate of purchase values, maximizing explained variance.
* The pipeline balances **accuracy, interpretability, and robustness**, making it suitable for real-world deployment.



## References

* [LightGBM Documentation](https://lightgbm.readthedocs.io/)
* [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
* Géron, A., *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, 2nd Edition
* Course materials from Machine Learning Practice, IIT Madras

