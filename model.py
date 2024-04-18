import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data from the text file
data_file = "data.txt"
data = np.loadtxt(data_file)

# Split data into features (X) and labels (y)
X = data[:, :-1]  # Features are all columns except the last one
y = data[:, -1]   # Labels are the last column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    shuffle=True,
                                                    stratify=y)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100,  # More trees (default is also 100 in newer versions)
                                       max_depth=None,    # None means nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
                                       min_samples_split=2,
                                       min_samples_leaf=1,
                                       random_state=42)   # Control the randomness for reproducibility

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_classifier.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

with open('./model', 'wb') as f:
    pickle.dump(rf_classifier, f)
