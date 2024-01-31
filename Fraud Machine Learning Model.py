#!/usr/bin/env python
# coding: utf-8

# In[10]:


#importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[11]:


#loading dataset
df = pd.read_csv('Fraud.csv')


# In[12]:


#first five rows of the dataset
df.head()


# In[13]:


#shape of the dataset
df.shape


# In[14]:


#information about the columns and their datatypes
df.info()


# In[15]:


df.columns


# In[18]:


#checking for any duplicate data
df[df.duplicated()]


# In[19]:


#checking for any null or misssing values
df.isnull().sum()


# In[20]:


#information regarding numerical columns
df.describe()


# In[21]:


#checking for outliers using box plot
for col in df.columns:
    if df[col].dtype == 'float64' or df[col].dtype == 'int64':
        print(col)
        df.boxplot(column = col)
        plt.show()
    else:
        pass


# In[22]:


#checking for outliers
numerical_columns = ['step','amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

# Initialize a dictionary to store the number of outliers for each column
outliers_count = {}

for col in numerical_columns:
    # Calculate the IQR for each numerical column
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    # Identify potential outliers using the IQR method
    outliers = ((df[col] < (Q1 - 1.5 * IQR)) |
                (df[col] > (Q3 + 1.5 * IQR)))
    
    # Count the number of outliers for the current column
    num_outliers = outliers.sum()
    
    # Store the count in the dictionary
    outliers_count[col] = num_outliers

# Display the number of outliers for each column
for col, count in outliers_count.items():
    print(f"Number of outliers in column '{col}': {count}")


# In[24]:


# I believe the reasons for these columns having outliers are legitimate and keeping them as it may provide any hidden patterns.


# In[25]:


sns.countplot(df["isFraud"])  # Zero Establish mean no fraud
df.isFraud.value_counts()      # One Establish mean fraud


# In[26]:


sns.countplot(df["isFlaggedFraud"]) # Zero Establish mean no fraud
df.isFlaggedFraud.value_counts()    # One Establish mean fraud


# In[27]:


#checking for multicollinearity
df.corr()


# In[28]:


correlation_matrix = df.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True)
plt.show()


# In[29]:


#We can see from the above heatmap that oldbalanceOrig and newbalanceOrig have collinearity of 1 we can't define the individual effects of these columns for the model and redundancy of having two columns with same information may lead to overfitting. So, adding a new feature called balance which is the difference between oldbalanceOrig and newbalanceOrig will help. Same with oldbalanceDest and newbalanceDest.


# In[30]:


# Create new columns for balance changes
df['balanceChangeOrig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
df['balanceChangeDest'] = df['newbalanceDest'] - df['oldbalanceDest']

# Drop the original balance columns
df.drop(['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'], axis=1, inplace=True)


# In[31]:


correlation_matrix = df.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True)
plt.show()


# In[32]:


df.type.value_counts()


# In[33]:


#Data encoding converting categorical columns to numerical
encoded_types = pd.get_dummies(df['type'], prefix='type')
df = pd.concat([df, encoded_types], axis=1)

# Drop the original 'type' column
df.drop(['type'], axis=1, inplace=True)


# In[34]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df['nameOrig'] = labelencoder.fit_transform(df['nameOrig'])
df['nameDest'] = labelencoder.fit_transform(df['nameDest'])


# In[35]:


from sklearn.model_selection import train_test_split

# Splitting the data into features (X) and target (y)
X = df.drop(['isFraud'], axis=1)  # Features
y = df['isFraud']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# In[36]:


from sklearn.preprocessing import StandardScaler
# Feature Scaling: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[37]:


from sklearn.linear_model import LogisticRegression
# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=500)
model.fit(X_train_scaled, y_train)


# In[39]:


model.score(X_train_scaled,y_train)


# In[40]:


model.score(X_test_scaled,y_test)


# In[41]:


from sklearn.tree import DecisionTreeClassifier
model_dt = DecisionTreeClassifier(random_state = 2)
model_dt.fit(X_train_scaled,y_train)


# In[42]:


model_dt.score(X_test_scaled,y_test)


# In[43]:


feature_importances = model_dt.feature_importances_

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
})

# Sort features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print the feature importance DataFrame
print(feature_importance_df)


# In[45]:


# Our fraud detection model is designed to identify potentially fraudulent transactions within a financial company's system. The model utilizes a machine learning algorithm called Decision Tree Classifier to predict whether a given transaction is likely to be fraudulent or not.
# The key factors in predicting fraudelent transactions are balanceChangeOrig, balanceChangeDest, step, type_TRANSFER, nameOrig, nameDest, amount, type_CASH_OUT, isFlaggedFraud, and type_PAYMENT. These factors make sense in the context of fraud detection for several reasons: 
# Balance Changes: Large and unusual balance changes in the accounts of both the transaction initiator and the recipient can indicate fraudulent behavior. step: The timing of transactions is important as fraudulent activities might occur during specific time periods. 
# 'TRANSFER' and 'CASH_OUT' transactions because they involve the movement of funds. 'nameOrig' and 'nameDest' are relevant since specific customers might be involved in repeated fraudulent activities. Amount: Unusually high transaction amounts can suggest fraud. 
# 'isFlaggedFraud' flag indicates that the transaction was flagged as potentially fraudulent. 'type_PAYMENT' feature's lower importance suggests that regular payment transactions have less predictive power for fraud detection


# In[ ]:


#What are the key factors that predict fraudulent customer?  


# In[ ]:


#Unusual Transaction Frequency:
Fraudulent customers may exhibit an unusually high or low transaction frequency compared to regular customers.

#Unusual Transaction Amounts:
Large or irregular transaction amounts that deviate significantly from a customer's typical spending behavior can be indicative of fraud.

#Multiple Account Access:
Frequent access to multiple accounts or a sudden change in login patterns may suggest fraudulent activity.

#Device Anomalies:
Suspicious logins or transactions from new or unfamiliar devices may indicate potential fraud.

#Abnormal Transaction Times:
Transactions occurring at unusual times, especially during non-business hours or holidays, may be considered suspicious.

#High-Risk Countries:
Transactions from countries with a high risk of fraudulent activities may be flagged.


# In[ ]:


#Do these factors make sense? If yes, How? If not, How not?  


# In[ ]:


#Device and Account Security:
Yes, because changes in devices or frequent access to multiple accounts can signal unauthorized access or account takeover, which are common in fraudulent activities.

#Abnormal Transaction Amounts:
Yes, because fraudulent transactions often involve unusual amounts that deviate from a customer's regular spending habits.

#Payment Method Changes:
Yes, because sudden or unexpected changes in payment methods, especially to high-risk ones, can be a sign of fraudulent activity.

#User Behavior Analysis:
Yes, because analyzing patterns like navigation, clicks, and session durations helps in detecting anomalies that might be associated with fraud.

#Social Network Analysis:
Yes, because fraudulent activities can often be linked to networks of related accounts. Analyzing social connections helps in identifying potential fraud rings.


# In[ ]:


#What kind of prevention should be adopted while company update its infrastructure? 


# In[ ]:


-Enforce the use of multi-factor authentication for accessing sensitive systems and data.
-Ensure that all software, operating systems, and applications are regularly updated with the latest security patches to address vulnerabilities.
-Implement encryption protocols for sensitive data both in transit and at rest to protect it from unauthorized access.
-Divide the network into segments to limit the impact of a security breach and prevent lateral movement by attackers.
-Provide regular training to employees on cybersecurity best practices, phishing awareness, and the importance of secure passwords.
-Enforce strict access controls, granting employees the minimum level of access necessary to perform their job functions (principle of least privilege).
-Implement continuous monitoring systems to detect and respond to suspicious activities in real-time.
-Develop and regularly test an incident response plan to efficiently address and mitigate security incidents.


# In[ ]:


# Assuming these actions have been implemented, how would you determine if they work? 


# In[ ]:


Evaluate the effectiveness of endpoint security measures by monitoring and analyzing any security incidents related to devices. Check if malware is being effectively detected and prevented.

Track the time it takes to report and resolve security incidents. A reduction in incident resolution time indicates improved efficiency.

Regularly review access control logs and permissions to ensure that employees have the necessary access levels and that there are no unauthorized accesses.

Test the backup and recovery procedures to ensure that data can be efficiently restored in case of data loss or a ransomware attack.
Evaluate the results of penetration testing to determine if identified vulnerabilities have been addressed and if new vulnerabilities have been introduced.

Ensure that the organization remains compliant with relevant industry regulations and standards. Non-compliance may indicate weaknesses in security measures.
Utilize UEBA solutions to analyze patterns of user behavior and identify anomalies that could indicate unauthorized or suspicious activities.

Collect feedback from employees who have undergone security awareness training to gauge the effectiveness of the training and identify areas for improvement.

