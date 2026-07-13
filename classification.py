# =====================================================================
# PROJECT 2: DATA CLASSIFICATION (SUPERVISED LEARNING)
# Powered by DecodeLabs
# =====================================================================

# --- IMPORTS ---
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ==========================================
# STEP 1: INPUT (Raw Material Loading)
# ==========================================
print("Loading Iris Benchmark Dataset...")
iris = load_iris()
X = iris.data    # Features: Sepal/Petal dimensions
y = iris.target  # Labels: Setosa, Versicolor, Virginica
print(f"Total Samples Loaded: {X.shape[0]}")

# ==========================================
# STEP 2: PROCESS (The Logic Skeleton)
# ==========================================
# A. Structural Integrity: Train-Test Split (80% Train, 20% Test)
# Data is shuffled using random_state=42 to remove order bias
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training Data: {X_train.shape[0]} | Testing Data: {X_test.shape[0]}")

# B. The Gatekeeper: Feature Scaling
scaler = StandardScaler()
# .fit_transform on training data so it learns the pattern (-2 to +2 range)
X_train_scaled = scaler.fit_transform(X_train) 
# .transform on test data (applying the same scale without cheating)
X_test_scaled = scaler.transform(X_test)       

# C. The Workflow: Instantiate, Fit, and Predict
# 1. Instantiate: Build the frame with Optimal K=5
model = KNeighborsClassifier(n_neighbors=5)

# 2. Fit: Train the model (Machine learns the logic)
print("Training the K-Nearest Neighbors Engine...")
model.fit(X_train_scaled, y_train)

# 3. Predict: Apply derived logic on unseen test data
predictions = model.predict(X_test_scaled)

# ==========================================
# STEP 3: OUTPUT (Validation & Diagnostics)
# ==========================================
print("\n==========================================")
print("--- MODEL DIAGNOSTICS & VALIDATION ---")
print("==========================================")

# Accuracy Check
accuracy = accuracy_score(y_test, predictions) * 100
print(f"Accuracy: {accuracy:.2f}%")

# F1 Score (Harmonic Mean of Precision and Recall)
f1 = f1_score(y_test, predictions, average='macro')
print(f"F1 Score: {f1:.4f}")

# Confusion Matrix (True Positives, False Positives, etc.)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("==========================================")