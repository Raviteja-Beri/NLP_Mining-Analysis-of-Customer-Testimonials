import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, HuberRegressor
from sklearn.svm import LinearSVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
import xgboost as xgb
import pickle
import joblib
import os

# Create a directory to save models
model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)

# --- 1. Data Loading ---
dataset = pd.read_csv(r"C:\Users\HP\Downloads\Restaurant_Reviews.tsv", delimiter='\t')

# --- 2. Text Preprocessing ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

corpus = []
lemmatizer = WordNetLemmatizer()
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# --- 3. Feature Extraction (TF-IDF Vectorization) ---
tfidf_vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
X = tfidf_vectorizer.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Save the TF-IDF vectorizer
pickle.dump(tfidf_vectorizer, open(os.path.join(model_dir, "tfidf_vectorizer.pkl"), "wb"))
joblib.dump(tfidf_vectorizer, os.path.join(model_dir, "tfidf_vectorizer.joblib"))

# --- 4. Splitting the Dataset ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Helper function to save models
def save_model(model, model_name):
    pickle.dump(model, open(os.path.join(model_dir, f"{model_name}.pkl"), "wb"))
    joblib.dump(model, os.path.join(model_dir, f"{model_name}.joblib"))

# Helper function to evaluate and save regression models
def evaluate_and_save_regression_model(model, X_train, y_train, X_test, y_test, model_name):
    print(f"\n--- {model_name} ---")
    model.fit(X_train, y_train)
    y_pred_reg = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_reg)
    mae = mean_absolute_error(y_test, y_pred_reg)
    r2 = r2_score(y_test, y_pred_reg)
    print(f"Mean Squared Error ({model_name}): {mse:.4f}")
    print(f"Mean Absolute Error ({model_name}): {mae:.4f}")
    print(f"R-squared ({model_name}): {r2:.4f}")
    y_pred_reg_binary = (y_pred_reg > 0.5).astype(int)
    cm_reg = confusion_matrix(y_test, y_pred_reg_binary)
    print(f"Confusion Matrix ({model_name} - Binary):\n", cm_reg)
    ac_reg = accuracy_score(y_test, y_pred_reg_binary)
    print(f"Accuracy Score ({model_name} - Binary): {ac_reg:.4f}")
    save_model(model, model_name.replace(" ", "_").lower())

# --- 5. Classification Models ---

# Logistic Regression
print("\n--- Logistic Regression Model ---")
classifier_lr = LogisticRegression(random_state=0, solver='liblinear', max_iter=1000)
classifier_lr.fit(X_train, y_train)
y_pred_lr = classifier_lr.predict(X_test)
cm_lr = confusion_matrix(y_test, y_pred_lr)
print("Confusion Matrix (Logistic Regression):\n", cm_lr)
ac_lr = accuracy_score(y_test, y_pred_lr)
print(f"Accuracy Score (Logistic Regression): {ac_lr:.4f}")
cv_scores_lr = cross_val_score(classifier_lr, X, y, cv=10)
print(f"Mean Cross-validation Accuracy (Logistic Regression): {cv_scores_lr.mean():.4f}")
save_model(classifier_lr, "logistic_regression")

# Hyperparameter Tuned Logistic Regression
print("\n--- Hyperparameter Tuning with GridSearchCV (Logistic Regression) ---")
param_grid_lr = {
    'C': [0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs'],
    'penalty': ['l1', 'l2']
}
grid_search_lr = GridSearchCV(LogisticRegression(random_state=0, max_iter=1000), param_grid_lr, cv=5, verbose=0, n_jobs=-1)
grid_search_lr.fit(X_train, y_train)
best_classifier_lr = grid_search_lr.best_estimator_
y_pred_best_lr = best_classifier_lr.predict(X_test)
cm_best_lr = confusion_matrix(y_test, y_pred_best_lr)
print("Confusion Matrix (Best Logistic Regression):\n", cm_best_lr)
ac_best_lr = accuracy_score(y_test, y_pred_best_lr)
print(f"Accuracy Score (Best Logistic Regression): {ac_best_lr:.4f}")
cv_scores_best_lr = cross_val_score(best_classifier_lr, X, y, cv=10)
print(f"Mean Cross-validation Accuracy (Best Logistic Regression): {cv_scores_best_lr.mean():.4f}")
save_model(best_classifier_lr, "best_logistic_regression")

# K-Nearest Neighbors Classifier
print("\n--- K-Nearest Neighbors (KNN) Classifier ---")
classifier_knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier_knn.fit(X_train, y_train)
y_pred_knn = classifier_knn.predict(X_test)
cm_knn = confusion_matrix(y_test, y_pred_knn)
print("Confusion Matrix (KNN):\n", cm_knn)
ac_knn = accuracy_score(y_test, y_pred_knn)
print(f"Accuracy Score (KNN): {ac_knn:.4f}")
cv_scores_knn = cross_val_score(classifier_knn, X, y, cv=10)
print(f"Mean Cross-validation Accuracy (KNN): {cv_scores_knn.mean():.4f}")
save_model(classifier_knn, "knn_classifier")

# Support Vector Machine (LinearSVC)
print("\n--- Support Vector Machine (SVM) - LinearSVC ---")
classifier_svc = LinearSVC(random_state=0, max_iter=1000)
classifier_svc.fit(X_train, y_train)
y_pred_svc = classifier_svc.predict(X_test)
cm_svc = confusion_matrix(y_test, y_pred_svc)
print("Confusion Matrix (LinearSVC):\n", cm_svc)
ac_svc = accuracy_score(y_test, y_pred_svc)
print(f"Accuracy Score (LinearSVC): {ac_svc:.4f}")
cv_scores_svc = cross_val_score(classifier_svc, X, y, cv=10)
print(f"Mean Cross-validation Accuracy (LinearSVC): {cv_scores_svc.mean():.4f}")
save_model(classifier_svc, "linear_svc")

# Naive Bayes (MultinomialNB)
print("\n--- Naive Bayes (MultinomialNB) Classifier ---")
classifier_mnb = MultinomialNB()
classifier_mnb.fit(X_train, y_train)
y_pred_mnb = classifier_mnb.predict(X_test)
cm_mnb = confusion_matrix(y_test, y_pred_mnb)
print("Confusion Matrix (MultinomialNB):\n", cm_mnb)
ac_mnb = accuracy_score(y_test, y_pred_mnb)
print(f"Accuracy Score (MultinomialNB): {ac_mnb:.4f}")
cv_scores_mnb = cross_val_score(classifier_mnb, X, y, cv=10)
print(f"Mean Cross-validation Accuracy (MultinomialNB): {cv_scores_mnb.mean():.4f}")
save_model(classifier_mnb, "multinomial_nb")

# Random Forest Classifier
print("\n--- Random Forest Classifier ---")
classifier_rf = RandomForestClassifier(n_estimators=100, random_state=0)
classifier_rf.fit(X_train, y_train)
y_pred_rf = classifier_rf.predict(X_test)
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix (RandomForestClassifier):\n", cm_rf)
ac_rf = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy Score (RandomForestClassifier): {ac_rf:.4f}")
cv_scores_rf = cross_val_score(classifier_rf, X, y, cv=10)
print(f"Mean Cross-validation Accuracy (RandomForestClassifier): {cv_scores_rf.mean():.4f}")
save_model(classifier_rf, "random_forest_classifier")

# Gradient Boosting Classifier
print("\n--- Gradient Boosting Classifier ---")
classifier_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=0)
classifier_gb.fit(X_train, y_train)
y_pred_gb = classifier_gb.predict(X_test)
cm_gb = confusion_matrix(y_test, y_pred_gb)
print("Confusion Matrix (GradientBoostingClassifier):\n", cm_gb)
ac_gb = accuracy_score(y_test, y_pred_gb)
print(f"Accuracy Score (GradientBoostingClassifier): {ac_gb:.4f}")
cv_scores_gb = cross_val_score(classifier_gb, X, y, cv=10)
print(f"Mean Cross-validation Accuracy (GradientBoostingClassifier): {cv_scores_gb.mean():.4f}")
save_model(classifier_gb, "gradient_boosting_classifier")

# --- 6. Regression Models ---

# Linear Regression
evaluate_and_save_regression_model(LinearRegression(), X_train, y_train, X_test, y_test, "Linear Regression")

# Ridge Regression
evaluate_and_save_regression_model(Ridge(alpha=1.0), X_train, y_train, X_test, y_test, "Ridge Regression")

# Lasso Regression
evaluate_and_save_regression_model(Lasso(alpha=0.01, max_iter=1000), X_train, y_train, X_test, y_test, "Lasso Regression")

# ElasticNet Regression
evaluate_and_save_regression_model(ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000), X_train, y_train, X_test, y_test, "ElasticNet Regression")

# SGD Regressor
evaluate_and_save_regression_model(SGDRegressor(max_iter=1000, tol=1e-3, random_state=0), X_train, y_train, X_test, y_test, "SGD Regressor")

# Huber Regressor
evaluate_and_save_regression_model(HuberRegressor(max_iter=1000), X_train, y_train, X_test, y_test, "Huber Regressor")

# Random Forest Regressor
evaluate_and_save_regression_model(RandomForestRegressor(n_estimators=100, random_state=0), X_train, y_train, X_test, y_test, "Random Forest Regressor")

# Support Vector Regressor (SVR)
evaluate_and_save_regression_model(SVR(kernel='rbf'), X_train, y_train, X_test, y_test, "Support Vector Regressor")

# Polynomial Regression
print("\n--- Polynomial Regression (Degree 2) ---")
poly_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('linear', LinearRegression())
])
try:
    evaluate_and_save_regression_model(poly_pipeline, X_train, y_train, X_test, y_test, "Polynomial Regression")
except Exception as e:
    print(f"Could not run Polynomial Regression due to: {e}. This often happens with high-dimensional data.")

# MLP Regressor
evaluate_and_save_regression_model(MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=0, solver='adam'), X_train, y_train, X_test, y_test, "MLP Regressor")

# K-Neighbors Regressor
evaluate_and_save_regression_model(KNeighborsRegressor(n_neighbors=5), X_train, y_train, X_test, y_test, "KNeighbors Regressor")

# LightGBM Regressor
evaluate_and_save_regression_model(lgb.LGBMRegressor(random_state=0), X_train, y_train, X_test, y_test, "LightGBM Regressor")

# XGBoost Regressor
evaluate_and_save_regression_model(xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=0), X_train, y_train, X_test, y_test, "XGBoost Regressor")

print(f"\nAll models have been saved in the '{model_dir}' directory as both .pkl and .joblib files.")