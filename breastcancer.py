import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. Load Data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 2. Pre-processing (Neural Networks need scaled data to work)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Split into Train/Test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# 4. Define the Deep Learning Model (Multi-Layer Perceptron)
# This creates a network with 2 hidden layers of 20 neurons each
print("Training the Neural Network... please wait.")
mlp = MLPClassifier(hidden_layer_sizes=(15, 15), max_iter=2000)
mlp.fit(X_train, y_train)

# 5. Get Results
predictions = mlp.predict(X_test)
print("-" * 30)
print(f"Deep Learning Accuracy: {accuracy_score(y_test, predictions)*100:.2f}%")
print("-" * 30)

# 6. Save a 'Training Sample' for your report
sample_data = X.head()
sample_data.to_csv("medical_training_sample.csv")
print("Saved training sample to 'medical_training_sample.csv'")
