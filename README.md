# CREDIT CARD FRAUD DETECTION
![image](https://user-images.githubusercontent.com/116062465/231133305-0696d3b3-ce64-4c0e-b1df-afb357abc931.png)

# Overview  
Credit card fraud is defined as a fraudulent transaction (payment) that is made using a credit or debit card by an unauthorized user.Credit cards are now the most preferred way for customers to transact either offline or online, due to the advancement in communication and electronic commerce systems.It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.  

# Business understanding  
Due to the advancement in communication and electronic commerce systems, proliferation and increase in the use of services such as e-commerce, tap and pay systems, online bills payment systems etc. As a consequence, fraudsters have also increased activities to attack transactions that are made using credit cards.Fraud associated with  transactions has increased significantly and fraud detection has become a challenging task because of the constantly changing nature and patterns of the fraudulent transactions.It is therefore crucial to implement effective and efficient mechanisms that can detect credit card fraud to protect users from financial loss.

## Main objective  
Due to the increase in fraudulent activities it has become essential for financial institutions and businesses to develop advanced fraud detection techniques to counter the threat of fraudulent credit card transactions and identity theft and keep losses to a minimum  

## Specific Objectives
- To identify and analyze patterns and correlations in the dataset to better understand the characteristics of fraudulent transactions.
- To check time of the day when most frauds are conducted and ages that are prone to credit card Frauds.
- To contribute to the development of more accurate and effective fraud detection models for financial transactions.
- To provide a useful tool for individuals and organizations to detect and prevent fraudulent financial transactions, thereby minimizing financial losses and improving financial security.
- To compare different ML models predictions to achieve highest accuracy.
- Advice the Company on best the strategy.

# Data understanding  
The dataset used for this project was acquired from [Kaggle](https://user-images.githubusercontent.com/116062465/231133305-0696d3b3-ce64-4c0e-b1df-afb357abc931.png) .This dataset contained financial transactions that had been simulated using a real-world financial transactions dataset and it had 23 columns and rows and the target variable(is fraud) was a binary indicator showing whether the transaction was fraudulent 1 or normal 0.An exploratory data analysis was performed on the training data to identify features that were correlated with fraudulent activities. Models were then developed using those features, and their predictive effectiveness was evaluated. The features present in the dataset were analyzed.These are some of the features present in the dataset.

- Index - Unique Identifier for each row
- Trans_date_trans_time - Transaction DateTime
- Cc_num - Credit Card Number of Customer
- Merchant - Merchant Name
- Category - Category of Merchant
- Amt - Amount of Transaction
- First - First Name of Credit Card Holder
- Last - Last Name of Credit Card Holder
- Gender - Gender of Credit Card Holder
- Street - Street Address of Credit Card Holder
- City - City of Credit Card Holder
- State - State of Credit Card Holder
- Zip - Zip of Credit Card Holder
- Lat - Latitude Location of Credit Card Holder
- Long - Longitude Location of Credit Card Holder
- City_pop - Credit Card Holder's City Population
- Job -Job of Credit Card Holder
- Dob - Date of Birth of Credit Card Holder
- Rans_num - Transaction Number
- Unix_time - UNIX Time of transaction
- Merch_lat - Latitude Location of Merchant
- Merch_long - Longitude Location of Merchant
- Is_fraud - Fraud Flag <--- Target Class
- Index Unique - Identifier for each row
- Trans_date_trans_time - Transaction DateTime

# Data Pre Processing  
- The columns were renamed to a proper and understandable way
- checking for features with high correlation with the target(Is Fraud) variable. 
- Dropping the irrelevant columns('Date', 'Longitude', 'Merchant Latitude', 'Merchant Longitude').
- Dealing with Categorical columns  and the numerical columns by label Encoding  and feature scaling  using MinMaxScaler() respectively.Feature scaling was to ensure all features are on the same scale improving the model performance.
- Handling Class Imbalance using SMOTE Since there were less cases of fraud in the dataset  so as to balance the dataset  increasing the representation of the minority class in the training data.  

# Modelling  
In coming up with the best model, the following approach was  taken:
- Fitting potential classification models such as Random Forest classifier, Logistic Model, decision tree and K Nearest Neighbors on the balanced training dataset.
- Hyperparameters tuning of the two best models i.e Random Forest and Decision Tree(taking into account prediction accuracy and recall).
- Comparing vanilla versions of the two best models with the tuned versions of the models and selecting the best model for Deployment.

In summary, the evaluation of the four models for predicting fraud in a dataset was conducted, namely Logistic Regression, Decision Trees, Random Forest, and KNN, with variation in ROC scores and recall values across the models. The best results were achieved by the Random Forest model, followed closely by the Decision Trees model. As a result, it is recommended that the Random Forest and Decision Trees models undergo further tuning, as the highest ROC scores and recall values were achieved by them. Further tuning of these models has the potential to improve their performance and make them even more suitable for predicting fraud in similar datasets.

To evaluate fraud detection models, the concepts of sensitivty/recall and precision are very important. Recall is True Positives/(True Positives + False Negatives), which measures how many fraud cases fly under the radar while precision (True Positives/(True Positives + False Positives)) evaluates how good the model is at generating as fewer false alarms as possible. For fraud detection, prioritize high recall to leave out as few fraud cases as possible while also having a relatively high precision because too many false alarms can also be a problem.

# Evaluation  
A randomforest classifier and Decision Tree were chosen as the two best models and compared against each other according to their performance on predicting new unseen data. in line with the business problem. The Decision Tree model was chosen as the model for deployment as it had an ROC of 98% and a recall of 90%.

# Minimum Viable product
## Short comings:
- Imbalanced dataset, with a small number of fraud cases relative to non fraud cases
- Limited model tuning due to computational power constraints

## Done to improve:
- More relevant features engineered to better capture the nuances of the data.
- Different models experimented including logistic regression,Random Forest and KNN to find the best model for the problem.
- Use anomalous deection to check for outliers in different columns and to determine other factors that contribute to faudulent activities.  

## Strengths
- Best performing model,Decision Tree Tuned achieved a high ROC score of 98% and a recall of 99% for non fraud cases and 89% for fraud cases including indicating good performance on both cases.
- proof of concept demonstrated for using machine learning to address the problem of credit card fraud detection.
- We had a test dataset to prove if the model's fraud predictions were similar to that of fraudulent activities in the test dataset.
- Overall, the MVP for the credit card fraud detection model is a promising starting point for future improvements.Further Data collection and feature engineering as well as experimentation with different models and hyperparameters can help improve the accuracy and robustness of the model. 

# Conclusions

- After the models were evaluated, the Decision Tree model was selected as the final model based on the highest performance on evaluation metrics. An ROC score of 99%, recall of 95%, and F1-score of 99% were achieved by the final model, indicating its ability to classify instances of fraud with high accuracy and low false positives and false negatives.
- The dataset was highly imbalanced, with a small percentage of fraudulent transactions. Fraudulent transactions tend to peak around `$300` and then at the `$800-$1000` range, and the amount of fraudulent transactions tends to be higher than that of legitimate transactions. 
- Correlation analysis indicates that age and location are also factors that have some correlation with fraudulent transactions.
- Fradulent transaction accur in smaller amount to avoid suspicions. 
- Data seemed to suggest that females and males were almost equally susceptible `(50%) to transaction fraud.
- Overall, the importance of using machine learning techniques for fraud detection in credit card transactions is highlighted by the findings from the analysis and the performance of the selected model. I
- mplementing this model can better identify and prevent fraudulent transactions, ultimately saving time and money for both the company and their customers.

# Recommendations 
- The implementation of measures to prevent fraudulent transactions, such as two-factor authentication, alerts for unusual account activity, and transaction limits for certain types of purchases, is recommended.
- Real-time monitoring is recommended to be implemented to identify and prevent fraudulent transactions as they occur, particularly during the late hours of the night.
- The prioritization of fraud detection in high-risk states such as New York(NY), Texas(TX) and Pennsylvania(PA) is recommended.
- Recommend Public education on financial literacy and how to keep private information to people aged between 30 - 60 as they more susceptible to fraud. 
- For every transaction that is flagged as fraudulent, a human element can be added to verify whether the transaction was done by calling the customer.


# Repository guide
- The data set used can be found [here](https://www.kaggle.com/datasets/kartik2112/fraud-detection?select=fraudTest.csv)
- The data report can be found [here](https://docs.google.com/document/d/13bjQ34r_e0_QOK4VrOd3s_MSTSW1FWB-59lFCNM_5DI/edit?usp=sharing)
- The notebook can be found [here](https://github.com/K-made/Creditcard-FraudBusters/blob/main/.ipynb_checkpoints/credit-cards-checkpoint.ipynb)
- The Presentation Slides can be found [here](https://www.canva.com/design/DAFfIULTcgs/CQ9ealhZ-N0MgHn8T3lqhQ/view?utm_content=DAFfIULTcgs&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink#2)
- Deployment link [here](https://flavianmiano-credit-card-fraud-detection-home-b1e6gv.streamlit.app/User)
