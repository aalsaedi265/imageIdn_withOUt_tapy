

import pickle #serilization to used in the api
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the MNIST dataset fetch_openml is used to load the MNIST dataset, which consists of 28x28 pixel images of handwritten digits
X, y = fetch_openml(name='mnist_784', version=1, return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize RandomForestClassifier to use all available CPU cores
clf = RandomForestClassifier(n_jobs=-1)

# Train the model
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))

# Optionally, save the trained model using pickle
with open('mnist_model.pkl', 'wb') as f:
    pickle.dump(clf, f)