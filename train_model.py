import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Load dataset manually (handle '?' as NaN)
df = pd.read_csv('dataset/processed.hungarian.data', header=None)

# Print the columns to see the structure
print("Columns:", df.columns)

# Set column names (if not already present in the dataset)
df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# Replace '?' with NaN for missing values
df.replace('?', pd.NA, inplace=True)

# Convert all data to numeric (since some columns may have been read as strings)
df = df.apply(pd.to_numeric, errors='ignore')

# Handle missing data - we will use SimpleImputer to replace missing values with the mean of the column
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df.drop('target', axis=1)))

# The target column should not be imputed, so we keep it separate
y = df['target']

# Features (X)
X = df_imputed

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt_model.predict(X_test))

# KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_acc = accuracy_score(y_test, knn_model.predict(X_test))

print(f"Decision Tree Accuracy: {dt_acc:.2f}")
print(f"KNN Accuracy: {knn_acc:.2f}")

# Save the better model
best_model = dt_model if dt_acc >= knn_acc else knn_model
joblib.dump(best_model, 'model/heart_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
