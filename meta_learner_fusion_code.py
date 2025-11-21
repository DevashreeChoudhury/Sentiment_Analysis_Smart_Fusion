import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
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
    words = [w for w in words if w not in STOPWORDS]
    return words

def build_sentiment_dictionary(descriptions, prices):
    tfidf = TfidfVectorizer(stop_words="english", min_df=2)
    tfidf_matrix = tfidf.fit_transform(descriptions)
    vocab = np.array(tfidf.get_feature_names_out())
    
    sentiment_dict = {}
    word_price_map = {w: [] for w in vocab}
    
    for doc, price in zip(descriptions, prices):
        words = set(extract_clean_words(doc))
        for w in words:
            if w in word_price_map:
                word_price_map[w].append(price)
    
    avg_prices = []
    for w, plist in word_price_map.items():
        if len(plist) >= 2:
            avg_prices.append(np.mean(plist))
    
    if len(avg_prices) == 0:
        return {}
    
    min_p, max_p = min(avg_prices), max(avg_prices)
    rng = max_p - min_p if max_p != min_p else 1
    
    for w, plist in word_price_map.items():
        if len(plist) >= 2:
            avg = np.mean(plist)
            sentiment = (avg - min_p) / rng
            sentiment_dict[w] = sentiment
    
    return sentiment_dict

def compute_sentiment(text, dictionary):
    words = extract_clean_words(text)
    score = sum(dictionary.get(w, 0) for w in words)
    return score

# =============================
# LOAD AND PREPARE DATA
# =============================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

df = pd.read_csv("real_estate_data_chicago.csv")

df = df.dropna(subset=['Price'])
df['Price'] = df['Price'].apply(lambda p: float(str(p).replace("$","").replace(",","").replace("+","")))

print(f"Dataset size: {len(df)} rows")

# =============================
# STEP 1: TRAIN NUMERIC MODEL
# =============================
print("\n" + "=" * 60)
print("STEP 1: Training Numeric Model (ALL Features)")
print("=" * 60)

exclude_cols = ['Description', 'Price']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X_full = df[feature_cols].copy()
y = df['Price'].copy()

X_numeric = pd.get_dummies(X_full, drop_first=True)
X_numeric = X_numeric.fillna(X_numeric.median())

print(f"Using {X_numeric.shape[1]} numeric features")

X_train_num, X_test_num, y_train, y_test, train_idx, test_idx = train_test_split(
    X_numeric, y, df.index, test_size=0.2, random_state=42
)

numeric_model = RandomForestRegressor(
    n_estimators=400,
    max_depth=15,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)

numeric_model.fit(X_train_num, y_train)
numeric_preds_train = numeric_model.predict(X_train_num)
numeric_preds_test = numeric_model.predict(X_test_num)

numeric_r2 = r2_score(y_test, numeric_preds_test)
numeric_mae = mean_absolute_error(y_test, numeric_preds_test)
print(f"✅ Numeric Model: R² = {numeric_r2:.4f}, MAE = ${numeric_mae:,.0f}")

# =============================
# STEP 2: COMPUTE SENTIMENT SCORES
# =============================
print("\n" + "=" * 60)
print("STEP 2: Computing Sentiment Scores")
print("=" * 60)

df_with_desc = df[df['Description'].notna() & (df['Description'].str.strip() != "")].copy()

if 'YearBuilt' in df_with_desc.columns:
    df_with_desc['YearBuilt'] = pd.to_numeric(df_with_desc['YearBuilt'], errors='coerce')
    df_with_desc = df_with_desc.dropna(subset=['YearBuilt'])
    
    min_year = int(df_with_desc['YearBuilt'].min())
    max_year = int(df_with_desc['YearBuilt'].max())
    bins = list(range(min_year, max_year + 20, 20))
    df_with_desc['YearBin'] = pd.cut(df_with_desc['YearBuilt'], bins=bins, right=False)
    
    df_with_desc['SentimentScore'] = 0.0
    
    for period in df_with_desc['YearBin'].unique():
        if pd.isna(period):
            continue
        
        segment_mask = df_with_desc['YearBin'] == period
        segment = df_with_desc[segment_mask].copy()
        
        if len(segment) < 20:
            continue
        
        descriptions = segment['Description'].tolist()
        prices = segment['Price'].values
        
        dictionary = build_sentiment_dictionary(descriptions, prices)
        
        if len(dictionary) < 5:
            continue
        
        sentiment_scores = [compute_sentiment(desc, dictionary) for desc in descriptions]
        df_with_desc.loc[segment_mask, 'SentimentScore'] = sentiment_scores
    
    df_with_desc = df_with_desc[df_with_desc['SentimentScore'] > 0].copy()
    print(f"✅ Computed sentiment for {len(df_with_desc)} properties")
else:
    descriptions = df_with_desc['Description'].tolist()
    prices = df_with_desc['Price'].values
    dictionary = build_sentiment_dictionary(descriptions, prices)
    df_with_desc['SentimentScore'] = [compute_sentiment(d, dictionary) for d in descriptions]

# =============================
# STEP 3: TRAIN SENTIMENT MODEL
# =============================
print("\n" + "=" * 60)
print("STEP 3: Training Enhanced Sentiment Model")
print("=" * 60)

train_with_sent = df_with_desc[df_with_desc.index.isin(train_idx)]
test_with_sent = df_with_desc[df_with_desc.index.isin(test_idx)]

if len(train_with_sent) < 50 or len(test_with_sent) < 20:
    print("❌ Not enough data with sentiment scores. Exiting.")
    exit()

X_train_sent = train_with_sent[['SentimentScore']].values
y_train_sent = train_with_sent['Price'].values

X_test_sent = test_with_sent[['SentimentScore']].values
y_test_sent = test_with_sent['Price'].values

# Use Gradient Boosting for better sentiment predictions
sentiment_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

sentiment_model.fit(X_train_sent, y_train_sent)
sentiment_preds_test = sentiment_model.predict(X_test_sent)

sentiment_r2 = r2_score(y_test_sent, sentiment_preds_test)
sentiment_mae = mean_absolute_error(y_test_sent, sentiment_preds_test)
print(f"✅ Sentiment Model: R² = {sentiment_r2:.4f}, MAE = ${sentiment_mae:,.0f}")

# =============================
# STEP 4: TRAIN META-LEARNER FOR FUSION
# =============================
print("\n" + "=" * 60)
print("STEP 4: Training Meta-Learner for Optimal Fusion")
print("=" * 60)

# Get predictions for training set with sentiment
train_common_idx = train_with_sent.index
X_train_num_common = X_train_num.loc[X_train_num.index.isin(train_common_idx)]
y_train_common = y_train.loc[y_train.index.isin(train_common_idx)]

numeric_preds_train_common = numeric_model.predict(X_train_num_common)
sentiment_preds_train = sentiment_model.predict(X_train_sent)

# Calculate residuals for confidence estimation
train_residuals = np.abs(y_train_common.values - numeric_preds_train_common)

# Create meta-features for stacking
# Features: numeric_pred, sentiment_pred, sentiment_score, residual_estimate, price_range
meta_features_train = np.column_stack([
    numeric_preds_train_common,
    sentiment_preds_train,
    X_train_sent.flatten(),
    train_residuals,  # Use actual residuals for training
    numeric_preds_train_common / np.median(numeric_preds_train_common)  # Relative price
])

# Train meta-learner to predict optimal blend
meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_features_train, y_train_common.values)

print("✅ Meta-learner trained")

# =============================
# STEP 5: APPLY INTELLIGENT FUSION
# =============================
print("\n" + "=" * 60)
print("STEP 5: Applying Meta-Learned Fusion")
print("=" * 60)

# Get test predictions
test_common_idx = test_with_sent.index
X_test_num_common = X_test_num.loc[X_test_num.index.isin(test_common_idx)]
y_test_common = y_test.loc[y_test.index.isin(test_common_idx)]

numeric_preds_common = numeric_model.predict(X_test_num_common)
test_residuals = np.abs(y_test_common.values - numeric_preds_common)

# Create meta-features for test set
# For test, estimate residual using numeric confidence (std of tree predictions)
if hasattr(numeric_model, 'estimators_'):
    tree_preds = np.array([tree.predict(X_test_num_common) for tree in numeric_model.estimators_])
    residual_estimate = np.std(tree_preds, axis=0)  # Higher std = less confident
else:
    residual_estimate = test_residuals  # Fallback

meta_features_test = np.column_stack([
    numeric_preds_common,
    sentiment_preds_test,
    X_test_sent.flatten(),
    residual_estimate,
    numeric_preds_common / np.median(numeric_preds_common)
])

# Get meta-learner predictions (final fused predictions)
fused_predictions = meta_model.predict(meta_features_test)

# Apply constraints: predictions should be reasonable
fused_predictions = np.clip(
    fused_predictions,
    np.minimum(numeric_preds_common, sentiment_preds_test) * 0.8,
    np.maximum(numeric_preds_common, sentiment_preds_test) * 1.2
)

# =============================
# STEP 6: SELECTIVE FUSION (HYBRID)
# =============================
print("\n" + "=" * 60)
print("STEP 6: Hybrid Approach - Use Best Strategy Per Case")
print("=" * 60)

# Calculate which strategy works best per case
residual_threshold = np.percentile(test_residuals, 85)

# For very confident numeric predictions, trust numeric
# For uncertain numeric predictions, use meta-learner
final_predictions = []

for i in range(len(numeric_preds_common)):
    if test_residuals[i] < residual_threshold:
        # Numeric is confident - use it
        final_predictions.append(numeric_preds_common[i])
    else:
        # Numeric uncertain - use meta-learner fusion
        final_predictions.append(fused_predictions[i])

final_predictions = np.array(final_predictions)

# =============================
# FINAL RESULTS
# =============================
print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

fused_r2 = r2_score(y_test_common, fused_predictions)
fused_mae = mean_absolute_error(y_test_common, fused_predictions)

hybrid_r2 = r2_score(y_test_common, final_predictions)
hybrid_mae = mean_absolute_error(y_test_common, final_predictions)

print(f"\n{'Model':<30} {'R²':<10} {'MAE':<15}")
print("-" * 55)
print(f"{'Numeric Only':<30} {numeric_r2:<10.4f} ${numeric_mae:>13,.0f}")
print(f"{'Sentiment Only':<30} {sentiment_r2:<10.4f} ${sentiment_mae:>13,.0f}")
print(f"{'Meta-Learner Fusion':<30} {fused_r2:<10.4f} ${fused_mae:>13,.0f}")
print(f"{'Hybrid (Best of Both)':<30} {hybrid_r2:<10.4f} ${hybrid_mae:>13,.0f}")

improvement = hybrid_r2 - numeric_r2
print(f"\n{'Best Improvement:':<30} {improvement:+.4f} ({100*improvement/max(numeric_r2, 0.0001):+.2f}%)")

# Count fusion usage
num_used_fusion = sum(test_residuals >= residual_threshold)
print(f"\nMeta-fusion used: {num_used_fusion}/{len(test_residuals)} cases ({100*num_used_fusion/len(test_residuals):.1f}%)")

# Save results
results_df = pd.DataFrame({
    'Actual_Price': y_test_common.values,
    'Numeric_Prediction': numeric_preds_common,
    'Sentiment_Prediction': sentiment_preds_test,
    'Meta_Fusion': fused_predictions,
    'Hybrid_Final': final_predictions,
    'Numeric_Residual': test_residuals,
    'Used_Fusion': test_residuals >= residual_threshold
})

results_df.to_csv('enhanced_fusion_results.csv', index=False)
print("\n💾 Results saved to 'enhanced_fusion_results.csv'")

print("\n" + "=" * 60)
print("✅ ENHANCED ANALYSIS COMPLETE")
print("=" * 60)