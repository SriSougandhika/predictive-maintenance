import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Load and preprocess data
data_path = 'ai2020.csv'
df1 = pd.read_csv(data_path)
print(df1.head())


def preprocess_data(df):
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with NaN values if needed, or fill them with a method of your choice
    df.dropna(inplace=True)

    # Feature engineering - Creating a new feature: Torque per rpm
    df['Torque per rpm'] = df['Torque [Nm]'] / df['Rotational speed [rpm]']
    print(df[['Torque per rpm']].head())

    # Convert 'Product ID' into quality variants by extracting the first letter
    df['Product_Quality'] = df['Product ID'].apply(lambda x: x[0])

    # Drop 'Product ID' since it's no longer needed
    df = df.drop(columns=['Product ID', 'UID'])

    # Label encode the categorical features
    label_encoder = LabelEncoder()
    df['Type'] = label_encoder.fit_transform(df['Type'])
    df['Product_Quality'] = label_encoder.fit_transform(df['Product_Quality'])

    # Features and target
    x = df.drop(columns=['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
    yscaled = df['Machine failure']  # Assuming we're predicting overall machine failure

    # Standardize the numerical features
    scaler = StandardScaler()
    xscaled = scaler.fit_transform(x)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    return xscaled, yscaled


X_scaled, y = preprocess_data(df1)
print(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
# Evaluate
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

# Save models using pickle
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(log_reg, f)

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
# Evaluate
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

# Support Vector Machine (SVC)
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
# Evaluate
print("SVC Accuracy:", accuracy_score(y_test, y_pred_svc))
print(classification_report(y_test, y_pred_svc))

with open('svc_model.pkl', 'wb') as f:
    pickle.dump(svc, f)

# Gradient Boosting Classifier
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
# Evaluate
print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))
print(classification_report(y_test, y_pred_gb))

with open('gradient_boosting_model.pkl', 'wb') as f:
    pickle.dump(gb, f)

# KNN classifier with 5 neighbors (you can adjust the number of neighbors)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
# Predict on the test set
y_pred_knn = knn.predict(X_test)
# Evaluate performance
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)

# Perform K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
# to find the optimal 'k':
wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

print(wcss)
print("Best value: ", max(wcss))

with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)
