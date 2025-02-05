import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('fraudTest.csv')

# Drop unnecessary columns
df.drop(columns=['Unnamed: 0', 'cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num'], inplace=True)

# Convert 'trans_date_trans_time' to datetime format
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])

# Extract datetime features
df['transaction_hour'] = df['trans_date_trans_time'].dt.hour
df['transaction_day'] = df['trans_date_trans_time'].dt.day
df['transaction_month'] = df['trans_date_trans_time'].dt.month
df['transaction_year'] = df['trans_date_trans_time'].dt.year

# Drop the original datetime column
df.drop(columns=['trans_date_trans_time'], inplace=True)

# Encode categorical variables using Label Encoding
label_encoders = {}
categorical_cols = ['merchant', 'category', 'gender', 'job']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders for future use

# Check class distribution (fraud vs. not fraud)
print(df['is_fraud'].value_counts())

# Separate majority and minority classes
fraud = df[df['is_fraud'] == 1]
legit = df[df['is_fraud'] == 0]

# Oversample the minority class (fraud)
fraud_upsampled = resample(fraud, replace=True, n_samples=len(legit), random_state=42)

# Combine the upsampled fraud data with legitimate data
df_balanced = pd.concat([legit, fraud_upsampled])

# Split the data into features (X) and target (y)
X = df_balanced.drop('is_fraud', axis=1)
y = df_balanced['is_fraud']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
