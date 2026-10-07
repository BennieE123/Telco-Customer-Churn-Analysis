\# Telco Customer Churn Analysis \& Prediction



\### An End-to-End Machine Learning Project for Customer Retention Analysis



\*\*Tools:\*\* Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost



\*\*Project Type:\*\* Exploratory Data Analysis, Predictive Modeling, Model Interpretation, and Business Analytics



\---



\## Project Overview



Customer churn is an important business problem for subscription-based companies because losing existing customers can affect revenue and increase the cost of acquiring new customers.



This project analyzes telecommunications customer data to identify patterns associated with customer churn and develops machine learning models to predict customers who may be at higher risk of churning.



The project follows an end-to-end analytical workflow:



\*\*Business Problem → Data Cleaning → Exploratory Data Analysis → Feature Preparation → Model Development → Model Evaluation → Model Interpretation → Business Recommendations\*\*



\---



\## Business Problem



The objective of this project is to understand customer churn patterns and determine whether customer characteristics can be used to predict churn.



The analysis addresses the following questions:



\* What proportion of customers churn?

\* Which customer characteristics are associated with higher observed churn rates?

\* How do contract type, tenure, internet service, payment method, and support services relate to churn?

\* Can machine learning models predict customers who are likely to churn?

\* Which features provide the most useful predictive information?

\* How does changing the classification threshold affect precision and recall?

\* How can the findings support customer-retention decisions?



\---



\## Dataset



The dataset contains \*\*7,043 customer records\*\* and \*\*21 variables\*\* describing customer demographics, services, contract information, billing characteristics, and churn status.



The target variable is \*\*Churn\*\*, indicating whether a customer left the service.



\### Target Distribution



\* Customers who did not churn: \*\*5,174\*\*

\* Customers who churned: \*\*1,869\*\*

\* Overall churn rate: \*\*26.54%\*\*



\### Data Quality



The analysis identified:



\* 11 missing `TotalCharges` values

\* No duplicate records

\* `customerID` treated as an identifier and excluded from modeling

\* `TotalCharges` converted to a numeric variable

\* Missing values handled through the modeling preprocessing pipeline



\---



\## Exploratory Data Analysis



Exploratory analysis was conducted at three levels:



1\. Univariate analysis

2\. Bivariate analysis

3\. Multivariate analysis



\### Customer Churn Distribution



!\[Customer Churn Distribution](images/churn\_distribution.png)



The dataset contains a larger proportion of customers who did not churn, while approximately \*\*26.54%\*\* of customers churned.



\---



\### Churn by Contract Type



!\[Churn by Contract Type](images/churn\_by\_contract.png)



Month-to-month customers exhibited substantially higher observed churn rates than customers on one-year and two-year contracts.



\---



\### Churn by Tenure



!\[Churn by Tenure](images/churn\_by\_tenure.png)



Customers with shorter tenure exhibited higher observed churn rates, with the highest churn concentrated among newer customers.



\---



\### Contract Type and Tenure



!\[Contract and Tenure Churn](images/contract\_tenure\_churn.png)



The interaction between contract type and tenure revealed a particularly strong pattern among month-to-month customers.



Observed churn among month-to-month customers declined across the tenure groups:



| Tenure Group | Churn Rate |

| ------------ | ---------: |

| 0–6 months   |     55.20% |

| 7–12 months  |     42.00% |

| 13–24 months |     37.72% |

| 25–48 months |     32.92% |

| 49–72 months |     26.02% |



These results describe associations within the dataset and should not be interpreted as evidence that contract type or tenure independently causes churn.



\---



\### Churn by Internet Service



!\[Churn by Internet Service](images/churn\_by\_internet\_service.png)



Observed churn differed across internet-service categories, with fiber-optic customers showing higher churn rates in this dataset.



The pattern was particularly pronounced among month-to-month customers and customers with shorter tenure.



This identifies a segment that may warrant further investigation into pricing, service quality, technical support, or customer experience.



\---



\## Machine Learning Approach



Five classification models were developed and evaluated:



1\. Logistic Regression

2\. Decision Tree

3\. Random Forest

4\. Gradient Boosting

5\. XGBoost



\### Feature Preparation



Numerical features:



\* `SeniorCitizen`

\* `tenure`

\* `MonthlyCharges`

\* `TotalCharges`



Categorical features included customer demographics, services, contract information, payment method, and other categorical variables.



\### Preprocessing



Numerical variables were:



\* Median-imputed

\* Standardized using `StandardScaler`



Categorical variables were:



\* Mode-imputed

\* One-hot encoded using `OneHotEncoder`



The preprocessing steps were incorporated into Scikit-learn pipelines to reduce the risk of data leakage.



\---



\## Train-Test Split



The dataset was divided using an \*\*80/20 stratified train-test split\*\*.



\* Training observations: \*\*5,634\*\*

\* Testing observations: \*\*1,409\*\*

\* Random state: \*\*42\*\*



Stratification was used to preserve the churn/non-churn distribution across the training and testing datasets.



\---



\## Model Performance



The models were evaluated using:



\* Accuracy

\* Precision

\* Recall

\* F1-score

\* ROC-AUC



ROC-AUC was given particular attention because churn prediction involves distinguishing between customers who churn and those who do not, while the business may operate at different classification thresholds.



| Model               | Accuracy | Precision | Recall | F1-score | ROC-AUC |

| ------------------- | -------: | --------: | -----: | -------: | ------: |

| Logistic Regression |   80.55% |    65.72% | 55.88% |   60.40% |  84.21% |

| Decision Tree       |   72.11% |    47.55% | 49.20% |   48.36% |  64.77% |

| Random Forest       |   78.35% |    61.86% | 48.13% |   54.14% |  82.06% |

| Gradient Boosting   |   80.27% |    66.55% | 51.60% |   58.13% |  84.33% |

| XGBoost             |   80.55% |    67.01% | 52.67% |   58.98% |  84.38% |



\### Model Comparison



!\[Model Comparison](images/model\_comparison.png)



The models produced different levels of predictive performance. Logistic Regression, Gradient Boosting, and XGBoost produced broadly similar ROC-AUC values, while the Decision Tree performed substantially worse.



\---



\## Overfitting Analysis



Training and testing accuracy were compared as a simple overfitting diagnostic.



| Model               | Training Accuracy | Test Accuracy | Accuracy Gap |

| ------------------- | ----------------: | ------------: | -----------: |

| Logistic Regression |            80.56% |        80.55% |        0.01% |

| Decision Tree       |            99.80% |        72.11% |       27.70% |

| Random Forest       |            99.80% |        78.35% |       21.45% |

| Gradient Boosting   |            82.96% |        80.27% |        2.69% |

| XGBoost             |            83.08% |        80.55% |        2.53% |



The Decision Tree showed substantial overfitting, while Gradient Boosting and XGBoost showed much smaller training-test gaps.



The accuracy gap is treated as a diagnostic rather than definitive proof of generalization.



\---



\## Cross-Validation



Five-fold cross-validation was used to evaluate the stability of model performance.



| Model               | Mean ROC-AUC | Standard Deviation |

| ------------------- | -----------: | -----------------: |

| Logistic Regression |       0.8462 |             0.0126 |

| Decision Tree       |       0.6534 |             0.0106 |

| Random Forest       |       0.8199 |             0.0126 |

| Gradient Boosting   |       0.8481 |             0.0126 |

| XGBoost             |       0.8472 |             0.0100 |



The three strongest models produced relatively similar cross-validation ROC-AUC values.



The small differences between their mean ROC-AUC values should not be interpreted as evidence that one model is substantially superior based on this dataset alone.



\---



\## Threshold Analysis



The default classification threshold of 0.50 was examined to understand how changing the threshold affects precision and recall.



!\[Threshold Analysis](images/threshold\_analysis.png)



| Threshold | Precision | Recall | F1-score |

| --------: | --------: | -----: | -------: |

|      0.30 |    52.50% | 75.67% |   61.99% |

|      0.40 |    58.78% | 64.44% |   61.48% |

|      0.50 |    67.01% | 52.67% |   58.98% |

|      0.60 |    73.71% | 38.24% |   50.35% |

|      0.70 |    78.63% | 24.60% |   37.47% |



Lowering the threshold from 0.50 to 0.30 increased recall from \*\*52.67% to 75.67%\*\*, but reduced precision from \*\*67.01% to 52.50%\*\*.



This demonstrates the trade-off between identifying more potential churners and limiting false-positive retention interventions.



In a production environment, the threshold should be selected using business costs, intervention capacity, and the relative cost of false positives and false negatives.



\---



\## Model Interpretation



\### Logistic Regression



Logistic Regression coefficients were examined to understand the direction of associations between model features and predicted churn probability.



Some larger positive coefficients included:



\* `InternetService\_Fiber optic`

\* `Contract\_Month-to-month`

\* `TotalCharges`

\* `StreamingMovies\_Yes`

\* `StreamingTV\_Yes`

\* `PaymentMethod\_Electronic check`



Some larger negative coefficients included:



\* `tenure`

\* `Contract\_Two year`

\* `InternetService\_DSL`

\* `MonthlyCharges`

\* `Dependents\_Yes`



These coefficients represent model associations rather than causal effects.



\---



\### XGBoost Feature Importance



XGBoost's built-in feature importance identified several prominent encoded features, including:



1\. `Contract\_Month-to-month`

2\. `InternetService\_Fiber optic`

3\. `TechSupport\_No`

4\. `OnlineSecurity\_No`

5\. `InternetService\_DSL`

6\. `StreamingMovies\_Yes`

7\. `tenure`



Because categorical variables were one-hot encoded, an original variable can be represented by multiple encoded features.



\---



\### Permutation Importance



Permutation importance was used to evaluate the effect of randomly shuffling individual original features on model ROC-AUC.



!\[Permutation Feature Importance](images/permutation\_importance.png)



| Feature         | Mean Importance |

| --------------- | --------------: |

| Contract        |          0.0918 |

| tenure          |          0.0607 |

| InternetService |          0.0092 |

| OnlineSecurity  |          0.0067 |

| MonthlyCharges  |          0.0061 |

| TechSupport     |          0.0051 |

| TotalCharges    |          0.0049 |



Contract and tenure showed substantially larger permutation importance than most other original features.



Near-zero or slightly negative permutation importance values should not automatically be interpreted as harmful features. Small negative values can occur because of sampling variation and correlations between predictors.



\---



\## Key Business Insights



\### 1. Contract Type



Month-to-month customers had substantially higher observed churn than customers on longer-term contracts.



\### 2. Customer Tenure



Newer customers showed higher observed churn, particularly within the month-to-month segment.



\### 3. Internet Service



Fiber-optic customers showed higher observed churn in several segments of the analysis.



\### 4. Payment Method



Electronic-check customers showed relatively high observed churn compared with several other payment methods.



\### 5. Support and Security Services



TechSupport and OnlineSecurity appeared among the more important predictive features, while customers without these services showed higher observed churn in several segments.



\### 6. Predictive Features



Permutation importance identified \*\*Contract\*\* and \*\*tenure\*\* as the strongest original features for the XGBoost model.



\---



\## Business Recommendations



\### 1. Strengthen Early-Tenure Retention



Develop structured onboarding and early engagement programs for new customers, particularly during the first six months.



\### 2. Monitor Month-to-Month Customers



Consider targeted engagement strategies for month-to-month customers, including loyalty initiatives, personalized offers, or contract-transition programs.



\### 3. Investigate Fiber-Optic Customer Experience



Investigate pricing, network performance, installation experience, service reliability, and technical support within the fiber-optic customer segment.



\### 4. Evaluate Electronic-Check Customers



Investigate whether the higher churn observed among electronic-check customers is related to other characteristics such as contract type, tenure, charges, or service usage.



\### 5. Use Predictive Risk Scores



Customer churn probabilities could be used to prioritize retention resources.



Customers can be segmented according to predicted risk so that limited retention resources are directed toward customers most likely to benefit from intervention.



\### 6. Select the Classification Threshold Based on Business Economics



The prediction threshold should be aligned with:



\* Cost of retention interventions

\* Cost of losing customers

\* Retention-team capacity

\* Expected intervention success rate

\* Cost of false positives and false negatives



\---



\## Limitations



Several limitations should be considered when interpreting the results.



\### Observational Data



The analysis identifies associations rather than causal relationships.



For example, higher observed churn among fiber-optic customers does not demonstrate that fiber-optic service causes churn.



\### Dataset-Specific Findings



The findings are based on the available dataset and may not generalize directly to other telecommunications companies, markets, or customer populations.



\### Model Imperfection



The models do not perfectly classify customer outcomes. Some customers will inevitably be misclassified.



\### Class Imbalance



The dataset contains more non-churn customers than churn customers. Therefore, accuracy alone is insufficient for evaluating model performance.



\### Threshold Selection



The appropriate classification threshold depends on business costs and operational capacity.



\### Production Validation



External validation on new customer data would be required before deploying the model in a real operational environment.



\---



\## Tools \& Technologies



\* \*\*Python\*\*

\* \*\*Pandas\*\*

\* \*\*NumPy\*\*

\* \*\*Matplotlib\*\*

\* \*\*Seaborn\*\*

\* \*\*Scikit-learn\*\*

\* \*\*XGBoost\*\*

\* \*\*Jupyter Notebook\*\*



\---



\## Project Structure



```text

Telco-Customer-Churn-Analysis/

│

├── data/

│   └── Telco-Customer-Churn.csv

│

├── images/

│   ├── churn\_distribution.png

│   ├── churn\_by\_contract.png

│   ├── churn\_by\_tenure.png

│   ├── contract\_tenure\_churn.png

│   ├── churn\_by\_internet\_service.png

│   ├── model\_comparison.png

│   ├── confusion\_matrix.png

│   ├── permutation\_importance.png

│   └── threshold\_analysis.png

│

├── notebooks/

│   └── Telco\_Customer\_Churn\_Analysis.ipynb

│

├── README.md

│

└── requirements.txt

```



\---



\## Conclusion



This project demonstrates an end-to-end approach to using data analysis and machine learning to understand customer churn.



The analysis combined exploratory data analysis, statistical interpretation, machine learning, model evaluation, cross-validation, threshold analysis, model interpretation, and business recommendations.



The results showed consistent associations between churn and factors such as contract type, tenure, internet service, payment method, and support-related services.



The machine learning models achieved ROC-AUC values around \*\*0.85\*\*, while permutation importance identified \*\*Contract\*\* and \*\*tenure\*\* as the strongest original features for the XGBoost model.



The project demonstrates how predictive analytics can move beyond model building to support practical business questions around customer retention, prioritization, and resource allocation.



\*\*Note:\*\* The findings represent associations in the analyzed dataset and should be validated with new data and controlled business experiments before being used for operational decision-making.



