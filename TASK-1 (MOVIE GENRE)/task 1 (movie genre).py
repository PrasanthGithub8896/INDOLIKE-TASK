import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset (replace 'path_to_dataset.csv' with your actual file path)
df = pd.read_csv("history.csv", encoding="latin-1")

# Check for missing data
print(df.isnull().sum())

# Fill or drop missing values
df = df.dropna(subset=['genre', 'rating'])  # Drop rows with missing genre or rating

# Clean 'runtime' column: Remove commas and extract the numeric part
df['runtime'] = df['runtime'].apply(lambda x: int(x.replace(',', '').split(' ')[0]) if isinstance(x, str) else x)

# Check if the runtime conversion worked
print(df['runtime'].head())

# Convert 'rating' to numeric, forcing errors to NaN and then handling them
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Handle missing ratings by filling with the mean (you can also drop rows with NaN ratings if you prefer)
df['rating'] = df['rating'].fillna(df['rating'].mean())

# One-hot encoding for 'genre' column (multi-genre column, so we'll create a column for each genre)
df_genres = df['genre'].str.get_dummies(sep=', ')
df = pd.concat([df, df_genres], axis=1)

# Select features for the model (you can add more features)
X = df[['rating', 'runtime'] + list(df_genres.columns)]  # Adding one-hot encoded genres as features

# Target variable: We'll predict if the movie is a 'Comedy' genre (binary classification for simplicity)
y = df['Comedy']  # For binary classification (Comedy or not)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model using accuracy score and classification report
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Cross-validation to evaluate model performance on different subsets of data
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Average Cross-Validation Score: {cv_scores.mean()}")

