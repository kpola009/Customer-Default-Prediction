# Customer Default Prediction

The Goal of this project is to predict how likely a customer is going to default, using historical customer data, This project can be identified as binary classification problem, which was solved using Logistic Regression and Random Forest.

## Introduction

Customer loan defaults refer to the failure of a borrower to repay a loan according to the agreed-upon terms. This can happen for a variety of reasons such as a change in financial circumstances. Loan defaults can have significant consequences for both the borrower and the lender. Understanding the causes of loan defaults and developing strategies to minimize them is a crucial aspect of responsible lending and credit risk management.

In this project as mentioned, machine learning techniques were leveraged to predict the likelihood of customer defaulting. By training this Machine learning models on historical data, lender can identify patterns and trends that are associated with higher risk default such as specific demographic characteristics or financial behaviors.

## Data Description

![alt text](https://github.com/kpola009/Customer-Default-Prediction/blob/main/Images/df.png "Figure 1")

Number of records: 94000

Number of features: 32

All the features in the dataset have numeric datatype and anonymized.

## EDA

- Initially, we started by analyzing the relationships between features in our dataset using a correlation matrix, since all of our features are numerical in nature. A correlation matrix is a useful tool in data analysis as it provides a graphical representation of the relationships between the numerical features. By computing the correlation coefficients between each pair of features, the matrix enables us to identify features that are highly correlated with one another, giving us valuable insights into the structure of the data. These insights can aid in feature selection and help us better understand the relationships between the features.

![alt text](https://github.com/kpola009/Customer-Default-Prediction/blob/main/Images/corr.png "Figure 2")

From the above corr matrix it was found, feature A22 and A24, A6 and A5, A10 and A9, A12 and A8 are highly correlated, by eliminating those features can result in better performing model.

- Secondly, histograms and probability plots were employed to investigate the normality of all the features in the dataset. The histograms provide a visual representation of the distribution of each feature, while the probability plots allow for a more precise assessment of normality by comparing the observed feature values to a theoretical normal distribution. These tools help us determine if the features in the dataset are normally distributed, which can be important in certain statistical tests and modeling techniques that assume normality. By using both histograms and probability plots, we can have a thorough understanding of the normality of the features in the dataset.

From the result of normality check it was found most of the features in dataset were highly skewed.

- Thirdly, boxplots were utilized to check for the presence of outliers in all the features. A boxplot provides a graphical representation of the distribution of a feature by plotting the median, quartiles, and any outliers.

From the boxplots, it was found most of the features contain outliers.

![alt text](https://github.com/kpola009/Customer-Default-Prediction/blob/main/Images/download.png "Figure 3")

Finally, using histogram it was determined that the target variable in our dataset is imbalanced. Specifically, there are more instances of customers who did not default compared to those who did default.

## Data Transformation

- Upon initial examination, it was found that there were no missing values in the dataset. However, a duplicate row was detected and was subsequently removed to maintain the accuracy of the data.
- Subsequently, it was observed that all features in the dataset were heavily skewed and contained outliers. To address this issue, the Yeo-Johnson transformation was applied to correct the skewness and handle the outliers.
- Thirdly, to handle imbalanced target variable, we will use StratifiedKFold cross validation technique during model training.

![alt text](https://github.com/kpola009/Customer-Default-Prediction/blob/main/Images/Drawing7%20(1).png "Figure 3")

Issues which were found during EDA were handled in this section as shown in above plots.

## ML Model Building
Given the nature of the problem, it was deemed a binary classification problem. To make predictions on the probability of customer default, both Logistic Regression and Random Forest models were utilized.

### a. Logistic Regression
The sklearn function gridSearchCV was utilized to find the optimal value for the hyperparameter C. It was determined that a value of 0.01 gave the best AUC score. Additionally, to address the imbalanced nature of the target variable, StratifiedKFolds was employed, resulting in an AUC score of 0.844515.

### b. Random Forest
The choice to use Random Forest was made because it is a tree-based algorithm, and it has a reputation for being capable of handling imbalanced data. Upon training the Random Forest algorithm, an AUC score of 0.81 was obtained.
