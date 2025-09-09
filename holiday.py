# Full pipeline cell: Preprocessing + Feature Engineering + SMOTE + Training + GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# --- imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42

# --- load data ---
df = pd.read_csv('travel (2).csv')

# --------------------------
# === Preprocessing (keeps your EDA style, made robust) ===
# --------------------------
# Fix some inconsistent labels you had
df['Gender'] = df['Gender'].replace({'Fe Male': 'Female'})
df['MaritalStatus'] = df['MaritalStatus'].replace({'Unmarried': 'Single'})

# Fill missing values (same logic you used, but safer)
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['TypeofContact'] = df['TypeofContact'].fillna('Unknown')
df['PreferredPropertyStar'] = df['PreferredPropertyStar'].fillna(df['PreferredPropertyStar'].mode().iloc[0])
df['NumberOfFollowups'] = df['NumberOfFollowups'].fillna(0)
df['NumberOfChildrenVisiting'] = df['NumberOfChildrenVisiting'].fillna(0)
df['NumberOfTrips'] = df['NumberOfTrips'].fillna(df['NumberOfTrips'].median())
df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
df['DurationOfPitch'] = df['DurationOfPitch'].fillna(df['DurationOfPitch'].median())

# Remove tiny-frequency occupation "Free Lancer" if present (like you did)
if 'Free Lancer' in df['Occupation'].unique():
    df = df[df['Occupation'] != 'Free Lancer']

# Map categories to numeric (keeping your mapping style)
df['TypeofContact'] = df['TypeofContact'].replace({'Self Enquiry': 0, 'Company Invited': 1, 'Unknown': 2})
df['Gender'] = df['Gender'].replace({'Male': 0, 'Female': 1})
df['Designation'] = df['Designation'].replace({'Manager':0,'Senior Manager':1,'Executive':2,'AVP':3,'VP':4})
df['ProductPitched'] = df['ProductPitched'].replace({'Basic':0,'Deluxe':1,'Standard':2,'Super Deluxe':3,'King':4})
df['MaritalStatus'] = df['MaritalStatus'].replace({'Single':0,'Married':1,'Divorced':2})
df['Occupation'] = df['Occupation'].replace({'Small Business':0,'Salaried':1,'Large Business':2})

# Drop ID and keep consistent columns
if 'CustomerID' in df.columns:
    df.drop(['CustomerID'], axis=1, inplace=True)

# Feature created in your EDA
df['Total_Visiting'] = df['NumberOfPersonVisiting'] + df['NumberOfChildrenVisiting']
df.drop(['NumberOfPersonVisiting', 'NumberOfChildrenVisiting'], axis=1, inplace=True)

# IQR capping (same method you used)
for col in ['NumberOfTrips', 'DurationOfPitch', 'MonthlyIncome']:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    IQR = q3 - q1
    upper = q3 + 1.5 * IQR
    lower = q1 - 1.5 * IQR
    df[col] = np.where(df[col] > upper, upper, np.where(df[col] < lower, lower, df[col]))

# You had dropped Gender previously; keep to match your notebook style (optional)
if 'Gender' in df.columns:
    df.drop('Gender', axis=1, inplace=True)

# --------------------------
# === Feature Engineering (to help push accuracy) ===
# --------------------------
# 1) Income per trip (avoid div by zero)
df['Income_per_Trip'] = df['MonthlyIncome'] / (df['NumberOfTrips'].replace(0, 1))
# 2) Income per person visiting
df['Income_per_Person'] = df['MonthlyIncome'] / (df['Total_Visiting'].replace(0, 1))
# 3) Pitch efficiency: income per second/minute pitched might be meaningful
df['Income_per_PitchDuration'] = df['MonthlyIncome'] / (df['DurationOfPitch'].replace(0, 1))
# 4) Interaction: CityTier * PreferredPropertyStar
df['Tier_Star_interaction'] = df['CityTier'] * df['PreferredPropertyStar']
# 5) Age bins (create small ordinal bucket)
df['Age_bin'] = pd.cut(df['Age'], bins=[15,25,35,45,60,100], labels=[0,1,2,3,4]).astype(int)

# Drop any accidental infinite or NaN from feature engineering
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# --------------------------
# === Prepare X, y and split ===
# --------------------------
X = df.drop(columns=['ProdTaken'])
y = df['ProdTaken']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    random_state=RANDOM_STATE, stratify=y)

# --------------------------
# === Scaling (you used MinMax for Age and StandardScaler for others) ===
# --------------------------
minmax = MinMaxScaler()
x_train['Age'] = minmax.fit_transform(x_train[['Age']])
x_test['Age'] = minmax.transform(x_test[['Age']])

std_d = StandardScaler()
x_train['DurationOfPitch'] = std_d.fit_transform(x_train[['DurationOfPitch']])
x_test['DurationOfPitch'] = std_d.transform(x_test[['DurationOfPitch']])

std_m = StandardScaler()
x_train['MonthlyIncome'] = std_m.fit_transform(x_train[['MonthlyIncome']])
x_test['MonthlyIncome'] = std_m.transform(x_test[['MonthlyIncome']])

# Also scale engineered numeric features to keep scale consistent
numeric_engineered = ['Income_per_Trip', 'Income_per_Person', 'Income_per_PitchDuration', 'Tier_Star_interaction']
scaler_eng = StandardScaler()
x_train[numeric_engineered] = scaler_eng.fit_transform(x_train[numeric_engineered])
x_test[numeric_engineered] = scaler_eng.transform(x_test[numeric_engineered])

# --------------------------
# === Balance the training set with SMOTE (keeps majority data) ===
# --------------------------
print("Before SMOTE class distribution:", y_train.value_counts().to_dict())
smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(x_train, y_train)
print("After SMOTE class distribution:", pd.Series(y_train_res).value_counts().to_dict())

# --------------------------
# === Evaluation utilities (fixed and reusable) ===
# --------------------------
def plot_cnf(model_name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def model_evaluation(model, ind_var, act, model_name="Model"):
    y_pred = model.predict(ind_var)
    # AUC if probabilities available
    try:
        y_proba = model.predict_proba(ind_var)[:,1]
        auc_test = roc_auc_score(act, y_proba)
    except Exception:
        auc_test = None
    cls = classification_report(act, y_pred, zero_division=0)
    accuracy = accuracy_score(act, y_pred)
    print(f"=== {model_name} ===")
    print("Accuracy:", round(accuracy, 4))
    if auc_test is not None:
        print("ROC AUC:", round(auc_test, 4))
    print("\nClassification Report:\n", cls)
    plot_cnf(model_name, act, y_pred)
    return accuracy

# --------------------------
# === Baseline models (train on SMOTE data) ===
baseline_models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'DecisionTree': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE),
    'AdaBoost': AdaBoostClassifier(random_state=RANDOM_STATE),
    'GradientBoosting': GradientBoostingClassifier(random_state=RANDOM_STATE)
}

baseline_results = {}
print("\nTraining baseline models (this may take a few seconds)...\n")
for name, mdl in baseline_models.items():
    mdl.fit(X_train_res, y_train_res)
    acc = model_evaluation(mdl, x_test, y_test, model_name=f"{name} (baseline)")
    baseline_results[name] = acc

# --------------------------
# === GridSearchCV tuning ===
# We expand RF grid moderately and tune AdaBoost (with base estimator depth)
# We also include small grids for DecisionTree & GradientBoosting & Logistic for completeness.
# --------------------------
print("\nRunning GridSearchCV for selected models (may take minutes)...\n")

# RandomForest Grid (expanded but practical)
# rf_param_grid = {
#     'n_estimators': [100, 300, 500],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5],
#     'min_samples_leaf': [1, 2],
#     'max_features': ['sqrt', 'log2'],
#     'bootstrap': [True]
# }
rf_param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [None, 15, 20],
    'max_features': ['sqrt', 0.8],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': [None, 'balanced']
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE),param_grid=rf_param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=2)

rf_grid.fit(X_train_res, y_train_res)
rf_best = rf_grid.best_estimator_
print("RF best params:", rf_grid.best_params_)
print("RF best CV score:", rf_grid.best_score_)


# Best model after tuning
best_rf = rf_grid.best_estimator_

# Predictions on test set
y_pred = best_rf.predict(x_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Test Accuracy:", acc)

# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# AdaBoost Grid (tune base estimator depth via base_estimator param objects)
ada_param_grid = {
    'estimator': [DecisionTreeClassifier(max_depth=1),
                       DecisionTreeClassifier(max_depth=2),
                       DecisionTreeClassifier(max_depth=3)],
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5, 1.0],
    'algorithm': ['SAMME.R']
}
ada_grid = GridSearchCV(AdaBoostClassifier(random_state=RANDOM_STATE),param_grid=ada_param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=2)
ada_grid.fit(X_train_res, y_train_res)
ada_best = ada_grid.best_estimator_
print("Ada best params:", ada_grid.best_params_)
print("Ada best CV score:", ada_grid.best_score_)

# Small grid for Decision Tree (optional)
dt_param_grid = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10]
}
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=RANDOM_STATE),
                       param_grid=dt_param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=0)
dt_grid.fit(X_train_res, y_train_res)
dt_best = dt_grid.best_estimator_

# Optional small grid for Logistic Regression (regularization)
lr_param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
}
lr_grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                       param_grid=lr_param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=0)
lr_grid.fit(X_train_res, y_train_res)
lr_best = lr_grid.best_estimator_

# --------------------------
# === Evaluate tuned models on test set ===
# --------------------------
print("\nEvaluation of tuned models on test set:\n")
rf_test_acc = model_evaluation(rf_best, x_test, y_test, model_name="RandomForest (tuned)")
ada_test_acc = model_evaluation(ada_best, x_test, y_test, model_name="AdaBoost (tuned)")
dt_test_acc = model_evaluation(dt_best, x_test, y_test, model_name="DecisionTree (tuned)")
lr_test_acc = model_evaluation(lr_best, x_test, y_test, model_name="LogisticRegression (tuned)")

# --------------------------
# === Final comparison table ===
# --------------------------
final_models = {
    'LogisticRegression_Tuned': lr_best,
    'DecisionTree_Tuned': dt_best,
    'RandomForest_Tuned': rf_best,
    'AdaBoost_Tuned': ada_best
}

comparison = []
for name, mdl in final_models.items():
    acc = accuracy_score(y_test, mdl.predict(x_test))
    comparison.append({'Model': name, 'Test Accuracy': round(acc, 4)})

comp_df = pd.DataFrame(comparison).sort_values(by='Test Accuracy', ascending=False).reset_index(drop=True)
print("\n=== Final comparison (test set) ===")
display(comp_df)

# Save best model (optional)
# import joblib
# joblib.dump(rf_best, 'rf_best_smote.joblib')
# joblib.dump(ada_best, 'ada_best_smote.joblib')

# If you want, display feature importances for the final RF
try:
    fi = pd.Series(rf_best.feature_importances_, index=x_train.columns).sort_values(ascending=False).head(20)
    print("\nTop feature importances (Random Forest):")

    display(fi)
    fi.plot(kind='barh', figsize=(8,6)); plt.gca().invert_yaxis(); plt.title('Top RF Feature Importances')
except Exception as e:
    print("Could not compute feature importances:", e)


#we have selected the final model : Random Forest
