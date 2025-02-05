import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Paths to dataset folders (UPDATE these paths)
cats_path = r"C:\Users\nivem\OneDrive\Desktop\INTERNSHIPS\Indolike\TASK 3 (Dog vs Cat)\cats"
dogs_path = r"C:\Users\nivem\OneDrive\Desktop\INTERNSHIPS\Indolike\TASK 3 (Dog vs Cat)\dogs"

# Function to load images and extract HOG features
def load_images_and_extract_features(folder_path, label):
    features = []
    labels = []
    
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
        img = cv2.resize(img, (128, 128))  # Resize image to 128x128
        
        # Extract HOG features
        hog_features, _ = hog(img, pixels_per_cell=(16, 16), 
                              cells_per_block=(2, 2), 
                              block_norm='L2-Hys', 
                              visualize=True)
        
        features.append(hog_features)
        labels.append(label)
    
    return np.array(features), np.array(labels)

# Load cat and dog images
cat_features, cat_labels = load_images_and_extract_features(cats_path, label=0)
dog_features, dog_labels = load_images_and_extract_features(dogs_path, label=1)

# Combine dataset
X = np.vstack((cat_features, dog_features))
y = np.hstack((cat_labels, dog_labels))

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train SVM model
svm = SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced')
svm.fit(X_train, y_train)

# Predictions
y_pred = svm.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
