import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder # Added this to handle the text labels
import joblib

# 1. Load the CLEANED dataset
print("Loading cleaned dataset...")
df = pd.read_csv('C:/Users/sushm/OneDrive/Desktop/iris_project/Cleaned_Iris.csv')

# 2. Separate features (X) and target (y)
X = df.drop('Species', axis=1).values
y_strings = df['Species'].values

# 3. Encode the string labels into integers and save the names
le = LabelEncoder()
y = le.fit_transform(y_strings)
target_names = le.classes_  # This captures ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# 4. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Initialize and train the KNN model (Distance Weighted)
print("Training Distance-Weighted KNN model...")
# weights='distance' makes closer neighbors have a greater influence
knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn_model.fit(X_train, y_train)

# 6. Evaluate the model (optional but good practice)
predictions = knn_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model trained successfully. Accuracy on test set: {accuracy * 100:.2f}%")

# 7. Save the trained model and the target names to disk
print("Saving model to model.pkl...")
# FIXED: Replaced 'iris.target_names' with our new 'target_names' variable
joblib.dump({'model': knn_model, 'target_names': target_names}, 'model.pkl')
print("Done!")