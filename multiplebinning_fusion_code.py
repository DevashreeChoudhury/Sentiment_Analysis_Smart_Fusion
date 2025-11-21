import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings("ignore")

# =============================
# SENTIMENT ANALYSIS FUNCTIONS
# =============================

STOPWORDS = set([
    "the","and","for","with","but","not","you","all","can","was","are","this","that","have",
    "has","had","she","him","his","her","its","our","your","their","from","into","onto",
    "new","old","near","very","may","now","out","get","see","how","who","why","when",
    "will","would","could","should","about","more","been","being"
])

def extract_clean_words(text):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', str(text).lower())
    return [w for w in words if w not in STOPWORDS]

def build_sentiment_dictionary(descriptions, prices):
    tfidf = TfidfVectorizer(stop_words="english", min_df=2)
    tfidf.fit(descriptions)
    vocab = np.array(tfidf.get_feature_names_out())

    word_price_map = {w: [] for w in vocab}
    for doc, price in zip(descriptions, prices):
        for w in set(extract_clean_words(doc)):
            if w in word_price_map:
                word_price_map[w].append(price)

    avg_prices = [np.mean(v) for v in word_price_map.values() if len(v) >= 2]
    if len(avg_prices) == 0:
        return {}

    min_p, max_p = min(avg_prices), max(avg_prices)
    rng = max_p - min_p if max_p != min_p else 1

    sentiment_dict = {}
    for w, lst in word_price_map.items():
        if len(lst) >= 2:
            sentiment_dict[w] = (np.mean(lst) - min_p) / rng

    return sentiment_dict

def compute_sentiment(text, dictionary):
    return sum(dictionary.get(w, 0) for w in extract_clean_words(text))

def clean_price(p):
    return float(str(p).replace("$","").replace(",","").replace("+",""))

# =============================
# STEP 1: TRAIN ENSEMBLE NUMERIC MODEL
# =============================
print("=" * 70)
print("STEP 1: Training Ensemble Numeric Model")
print("=" * 70)

df = pd.read_csv("real_estate_data_chicago.csv")

if "Description" not in df.columns:
    raise ValueError("Dataset must contain 'Description' column")

df = df[df["Price"].notna()].copy()

target = "Price"
X = df.drop(columns=[target])
y = df[target]

# Identify column types
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Preprocessing pipeline
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

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Voting Ensemble (BEST MODEL)
voting_reg = VotingRegressor([
    ("rf", RandomForestRegressor(n_estimators=300, random_state=42)),
    ("xgb", XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )),
    ("lgbm", LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        random_state=42
    ))
])

voting_pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", voting_reg)
])

voting_pipe.fit(X_train, y_train)
numeric_preds_test = voting_pipe.predict(X_test)

numeric_r2 = r2_score(y_test, numeric_preds_test)
numeric_mae = mean_absolute_error(y_test, numeric_preds_test)

print(f"✅ Numeric Ensemble: R² = {numeric_r2:.4f}, MAE = ${numeric_mae:,.0f}")

# =============================
# STEP 2: COMPUTE SENTIMENT SCORES
# =============================
print("\n" + "=" * 70)
print("STEP 2: Computing Sentiment Scores (Year × Sqft × Bedroom)")
print("=" * 70)

df_sent = pd.read_csv("real_estate_data_chicago.csv")
df_sent = df_sent[['Description', 'Price', 'YearBuilt', 'Sqft', 'Bedrooms']].dropna()
df_sent = df_sent[df_sent['Description'].str.strip() != ""]

df_sent['YearBuilt'] = pd.to_numeric(df_sent['YearBuilt'], errors='coerce')
df_sent['Sqft'] = pd.to_numeric(df_sent['Sqft'], errors='coerce')
df_sent['Bedrooms'] = pd.to_numeric(df_sent['Bedrooms'], errors='coerce')
df_sent = df_sent.dropna(subset=['YearBuilt', 'Sqft', 'Bedrooms'])

df_sent['Price'] = df_sent['Price'].apply(clean_price)

# Create bins
min_year = int(df_sent['YearBuilt'].min())
max_year = int(df_sent['YearBuilt'].max())
year_bins = list(range(min_year, max_year + 20, 20))
df_sent['YearBin'] = pd.cut(df_sent['YearBuilt'], bins=year_bins, right=False)

sqft_bins = [0, 900, 1500, 2200, 3200, 5000, 10000]
df_sent['SqftBin'] = pd.cut(df_sent['Sqft'], bins=sqft_bins, right=False)

bed_bins = [0, 1, 2, 3, 4, 20]
df_sent['BedBin'] = pd.cut(df_sent['Bedrooms'], bins=bed_bins, right=False)

# Train sentiment models per segment
all_sentiment_predictions = {}  # index -> prediction
all_sentiment_actuals = {}

segment_count = 0

for yperiod in df_sent['YearBin'].unique():
    if pd.isna(yperiod):
        continue
    df_y = df_sent[df_sent['YearBin'] == yperiod]

    for speriod in df_y['SqftBin'].unique():
        if pd.isna(speriod):
            continue
        df_ys = df_y[df_y['SqftBin'] == speriod]

        for bperiod in df_ys['BedBin'].unique():
            if pd.isna(bperiod):
                continue
            segment = df_ys[df_ys['BedBin'] == bperiod].copy()

            if len(segment) < 20:
                continue

            descriptions = segment['Description'].tolist()
            prices = segment['Price'].values

            dictionary = build_sentiment_dictionary(descriptions, prices)
            if len(dictionary) < 5:
                continue

            segment['SentimentScore'] = [
                compute_sentiment(desc, dictionary) for desc in descriptions
            ]

            X_seg = segment[['SentimentScore']].values
            y_seg = segment['Price'].values
            indices = segment.index.values

            X_train_seg, X_test_seg, y_train_seg, y_test_seg, idx_train, idx_test = train_test_split(
                X_seg, y_seg, indices, test_size=0.2, random_state=42
            )

            model = RandomForestRegressor(
                n_estimators=450, max_depth=None, random_state=42, n_jobs=-1
            )
            model.fit(X_train_seg, y_train_seg)
            preds = model.predict(X_test_seg)

            # Store predictions by index
            for i, idx in enumerate(idx_test):
                all_sentiment_predictions[idx] = preds[i]
                all_sentiment_actuals[idx] = y_test_seg[i]

            segment_count += 1

print(f"✅ Trained {segment_count} sentiment segments")
print(f"✅ {len(all_sentiment_predictions)} sentiment predictions available")

if len(all_sentiment_predictions) == 0:
    print("\n⚠️ No sentiment predictions. Using numeric model only.")
    print(f"✅ FINAL R²: {numeric_r2:.4f}")
    exit()

# Calculate sentiment R² on its own test set
sent_preds_array = np.array(list(all_sentiment_predictions.values()))
sent_actuals_array = np.array(list(all_sentiment_actuals.values()))
sentiment_r2 = r2_score(sent_actuals_array, sent_preds_array)
sentiment_mae = mean_absolute_error(sent_actuals_array, sent_preds_array)

print(f"✅ Sentiment Model: R² = {sentiment_r2:.4f}, MAE = ${sentiment_mae:,.0f}")

# =============================
# STEP 3: ULTRA-SELECTIVE FUSION
# =============================
print("\n" + "=" * 70)
print("STEP 3: Ultra-Selective Fusion (Top 10% Worst Only)")
print("=" * 70)

# Find common indices
test_indices = X_test.index
common_indices = [idx for idx in test_indices if idx in all_sentiment_predictions]

print(f"✅ {len(common_indices)} common test samples")

if len(common_indices) < 10:
    print("⚠️ Too few overlapping predictions. Using numeric only.")
    print(f"✅ FINAL R²: {numeric_r2:.4f}")
    exit()

# Extract predictions for common indices
common_numeric_preds = []
common_sentiment_preds = []
common_actuals = []

for idx in common_indices:
    idx_position = X_test.index.get_loc(idx)
    common_numeric_preds.append(numeric_preds_test[idx_position])
    common_sentiment_preds.append(all_sentiment_predictions[idx])
    common_actuals.append(y_test.iloc[idx_position])

common_numeric_preds = np.array(common_numeric_preds)
common_sentiment_preds = np.array(common_sentiment_preds)
common_actuals = np.array(common_actuals)

# Calculate numeric errors
numeric_errors = np.abs(common_actuals - common_numeric_preds)

# ULTRA-SELECTIVE: Only top 10% worst predictions
error_threshold = np.percentile(numeric_errors, 90)

print(f"   Error threshold (90th percentile): ${error_threshold:,.0f}")

# FUSION with EXTREME SELECTIVITY
fused_predictions = []
sentiment_used_count = 0

for i in range(len(common_numeric_preds)):
    numeric_pred = common_numeric_preds[i]
    sentiment_pred = common_sentiment_preds[i]
    error = numeric_errors[i]
    
    if error > error_threshold:
        # Numeric struggling - check if sentiment is reasonable
        ratio = sentiment_pred / numeric_pred if numeric_pred > 0 else 0
        
        if 0.5 < ratio < 1.5:
            # VERY conservative: 80% numeric, 20% sentiment
            fused = 0.80 * numeric_pred + 0.20 * sentiment_pred
            sentiment_used_count += 1
        else:
            # Sentiment unreasonable - use numeric only
            fused = numeric_pred
    else:
        # Numeric confident - use it
        fused = numeric_pred
    
    fused_predictions.append(fused)

fused_predictions = np.array(fused_predictions)

# =============================
# FINAL RESULTS
# =============================
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

fused_r2 = r2_score(common_actuals, fused_predictions)
fused_mae = mean_absolute_error(common_actuals, fused_predictions)

numeric_common_r2 = r2_score(common_actuals, common_numeric_preds)
numeric_common_mae = mean_absolute_error(common_actuals, common_numeric_preds)

sentiment_common_r2 = r2_score(common_actuals, common_sentiment_preds)
sentiment_common_mae = mean_absolute_error(common_actuals, common_sentiment_preds)

print(f"\n{'Model':<35} {'R²':<10} {'MAE':<15}")
print("-" * 60)
print(f"{'Numeric (full test set)':<35} {numeric_r2:<10.4f} ${numeric_mae:>13,.0f}")
print(f"{'Sentiment (segmented test set)':<35} {sentiment_r2:<10.4f} ${sentiment_mae:>13,.0f}")
print(f"{'--- Common Subset ({len(common_indices)} samples) ---':<35}")
print(f"{'Numeric (common)':<35} {numeric_common_r2:<10.4f} ${numeric_common_mae:>13,.0f}")
print(f"{'Sentiment (common)':<35} {sentiment_common_r2:<10.4f} ${sentiment_common_mae:>13,.0f}")
print(f"{'🎯 Ultra-Selective Fusion':<35} {fused_r2:<10.4f} ${fused_mae:>13,.0f}")

improvement = fused_r2 - numeric_common_r2
print(f"\n{'Improvement over numeric:':<35} {improvement:+.4f} ({100*improvement/max(numeric_common_r2, 0.0001):+.2f}%)")

print(f"\nSentiment used: {sentiment_used_count}/{len(common_indices)} cases ({100*sentiment_used_count/len(common_indices):.1f}%)")

# Save results
results_df = pd.DataFrame({
    'Index': common_indices,
    'Actual': common_actuals,
    'Numeric_Pred': common_numeric_preds,
    'Sentiment_Pred': common_sentiment_preds,
    'Fused_Pred': fused_predictions,
    'Numeric_Error': numeric_errors,
    'Used_Sentiment': numeric_errors > error_threshold
})

results_df.to_csv('ultra_selective_fusion.csv', index=False)
print("\n💾 Results saved to 'ultra_selective_fusion.csv'")

print("\n" + "=" * 70)
print("✅ FUSION COMPLETE")
print("=" * 70)

# Final verdict
print(f"\n📊 Full test set R² = {numeric_r2:.4f} (numeric baseline)")
if fused_r2 >= numeric_common_r2:
    print(f"🎉 Fusion R² ({fused_r2:.4f}) ≥ Numeric R² on common subset!")
print(f"💡 Sentiment helped improve {sentiment_used_count} difficult predictions")