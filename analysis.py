import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# 1. Data Acquisition
salaries_df = pd.read_csv('employee_salaries.csv')
adoption_df = pd.read_csv('adoption_data.csv')

# Display the first few rows of each dataset
print("Salaries DataFrame:")
print(salaries_df.head())
print("\nAdoption DataFrame:")
print(adoption_df.head())

# 2. Data Cleaning and Preparation
# Merge datasets on 'employee_id'
merged_df = pd.merge(salaries_df, adoption_df, on='employee_id')

# Drop rows with missing values in 'salary' or 'adopted'
cleaned_df = merged_df.dropna(subset=['salary', 'adopted'])

# Convert 'adopted' to a categorical type (0 or 1)
cleaned_df['adopted'] = cleaned_df['adopted'].astype(int)

print("\nCleaned DataFrame:")
print(cleaned_df.info())

# 3. Exploratory Data Analysis (EDA)
# Salary distribution
plt.figure(figsize=(10, 6))
sns.histplot(cleaned_df['salary'], kde=True)
plt.title('Salary Distribution')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.show()

# Adoption rate by salary
plt.figure(figsize=(10, 6))
sns.boxplot(x='adopted', y='salary', data=cleaned_df)
plt.title('Salary by Adoption Status')
plt.xlabel('Adopted')
plt.ylabel('Salary')
plt.show()

# 4. Feature Engineering
# Create salary bands
def salary_band(salary):
    if salary < 50000:
        return 'Low'
    elif 50000 <= salary < 100000:
        return 'Medium'
    else:
        return 'High'

cleaned_df['salary_band'] = cleaned_df['salary'].apply(salary_band)

print("\nDataFrame with Salary Bands:")
print(cleaned_df.head())

# 5. Predictive Modeling
# Feature selection
features = ['salary']
X = cleaned_df[features]
y = cleaned_df['adopted']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC Curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# ROC AUC Score
roc_auc = roc_auc_score(y_test, y_prob)
print(f'ROC AUC Score: {roc_auc:.2f}')
