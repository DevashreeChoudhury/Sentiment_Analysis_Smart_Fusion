# =============================
# 1) IMPORT LIBRARIES
# =============================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, VotingRegressor

from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import warnings
warnings.filterwarnings("ignore")


# =============================
# 2) LOAD DATA
# =============================
df = pd.read_csv("real_estate_data_chicago.csv")

# Drop "Description" column if needed (already named correctly)
if "Description" not in df.columns:
    raise ValueError("Dataset must contain 'Description' column")

df = df[df["Price"].notna()].copy()


# =============================
# 3) SET TARGET + FEATURES
# =============================
target = "Price"
X = df.drop(columns=[target])
y = df[target]


# =============================
# 4) IDENTIFY COLUMN TYPES
# =============================
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns


# =============================
# 5) FIX: IMPUTATION + SCALING PIPELINE
# =============================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])


# =============================
# 6) MODELS
# =============================
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="reg:squarederror"
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1
    )
}


# =============================
# 7) SPLIT DATA
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =============================
# 8) TRAIN ALL MODELS + R²
# =============================
scores = {}

for name, model in models.items():
    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])
    
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    r2 = r2_score(y_test, preds)
    
    scores[name] = r2
    print(f"{name}: R² = {r2:.4f}")


# =============================
# 9) ENSEMBLE (VOTING REGRESSOR)
# =============================
voting_reg = VotingRegressor([
    ("rf", models["Random Forest"]),
    ("xgb", models["XGBoost"]),
    ("lgbm", models["LightGBM"])
])

voting_pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", voting_reg)
])

voting_pipe.fit(X_train, y_train)
voting_preds = voting_pipe.predict(X_test)
voting_r2 = r2_score(y_test, voting_preds)

scores["Voting Ensemble"] = voting_r2

print("\nVoting Ensemble R² =", round(voting_r2, 4))


# =============================
# 10) PRINT ALL SCORES
# =============================
print("\n==============================")
print("   FINAL MODEL PERFORMANCE")
print("==============================")

for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{k}: R² = {v:.4f}")

print("\n✅ BEST MODEL:", max(scores, key=scores.get))
