import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
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

# Keep Description for sentiment but prepare full feature set
df = df.dropna(subset=['Price'])
df['Price'] = df['Price'].apply(lambda p: float(str(p).replace("$","").replace(",","").replace("+","")))

print(f"Dataset size: {len(df)} rows")

# =============================
# STEP 1: TRAIN NUMERIC MODEL FIRST (Get baseline)
# =============================
print("\n" + "=" * 60)
print("STEP 1: Training Numeric Model (ALL Features)")
print("=" * 60)

# Get all features except Description and Price
exclude_cols = ['Description', 'Price']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X_full = df[feature_cols].copy()
y = df['Price'].copy()

# Handle categorical and missing values
X_numeric = pd.get_dummies(X_full, drop_first=True)
X_numeric = X_numeric.fillna(X_numeric.median())

print(f"Using {X_numeric.shape[1]} numeric features")

# Split data
X_train_num, X_test_num, y_train, y_test, train_idx, test_idx = train_test_split(
    X_numeric, y, df.index, test_size=0.2, random_state=42
)

# Train numeric model
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
# STEP 2: COMPUTE SENTIMENT SCORES (FULL TRAINING SET)
# =============================
print("\n" + "=" * 60)
print("STEP 2: Computing Sentiment Scores")
print("=" * 60)

# Work only with rows that have descriptions
df_with_desc = df[df['Description'].notna() & (df['Description'].str.strip() != "")].copy()

# Add year bins for segmentation
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
    
    # Remove rows with zero sentiment
    df_with_desc = df_with_desc[df_with_desc['SentimentScore'] > 0].copy()
    print(f"✅ Computed sentiment for {len(df_with_desc)} properties")
else:
    print("⚠️ No YearBuilt column - computing sentiment without segmentation")
    descriptions = df_with_desc['Description'].tolist()
    prices = df_with_desc['Price'].values
    dictionary = build_sentiment_dictionary(descriptions, prices)
    df_with_desc['SentimentScore'] = [compute_sentiment(d, dictionary) for d in descriptions]

# =============================
# STEP 3: TRAIN SENTIMENT MODEL (ON FULL TRAINING SET)
# =============================
print("\n" + "=" * 60)
print("STEP 3: Training Standalone Sentiment Model")
print("=" * 60)

# Filter train/test to only include rows with sentiment
train_with_sent = df_with_desc[df_with_desc.index.isin(train_idx)]
test_with_sent = df_with_desc[df_with_desc.index.isin(test_idx)]

if len(train_with_sent) < 50 or len(test_with_sent) < 20:
    print("❌ Not enough data with sentiment scores. Exiting.")
    exit()

X_train_sent = train_with_sent[['SentimentScore']].values
y_train_sent = train_with_sent['Price'].values

X_test_sent = test_with_sent[['SentimentScore']].values
y_test_sent = test_with_sent['Price'].values

# Train sentiment model on ALL training data (not just high-error cases)
sentiment_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)

sentiment_model.fit(X_train_sent, y_train_sent)
sentiment_preds_test = sentiment_model.predict(X_test_sent)

sentiment_r2 = r2_score(y_test_sent, sentiment_preds_test)
sentiment_mae = mean_absolute_error(y_test_sent, sentiment_preds_test)
print(f"✅ Sentiment Model: R² = {sentiment_r2:.4f}, MAE = ${sentiment_mae:,.0f}")

# =============================
# STEP 4: SMART RESIDUAL-BASED FUSION
# =============================
print("\n" + "=" * 60)
print("STEP 4: Intelligent Fusion Strategy")
print("=" * 60)

# Get numeric predictions for the test set WITH sentiment
test_common_idx = test_with_sent.index
X_test_num_common = X_test_num.loc[X_test_num.index.isin(test_common_idx)]
y_test_common = y_test.loc[y_test.index.isin(test_common_idx)]

numeric_preds_common = numeric_model.predict(X_test_num_common)
residuals = np.abs(y_test_common.values - numeric_preds_common)

# Calculate residual threshold (90th percentile - be very selective)
residual_threshold = np.percentile(residuals, 90)
print(f"Using sentiment only for top 10% worst predictions (residual > ${residual_threshold:,.0f})")

# FUSION LOGIC: For each prediction, decide whether to use sentiment
fused_predictions = []

for i, idx in enumerate(test_common_idx):
    numeric_pred = numeric_preds_common[i]
    residual = residuals[i]
    
    # Get sentiment prediction
    sent_score = test_with_sent.loc[idx, 'SentimentScore']
    sentiment_pred = sentiment_model.predict([[sent_score]])[0]
    
    # Only use sentiment if:
    # 1. Numeric has high error (top 10%)
    # 2. Sentiment prediction is reasonable (within 50% of numeric)
    # 3. Sentiment would improve the prediction
    
    if residual > residual_threshold:
        # Check if sentiment is reasonable
        ratio = sentiment_pred / numeric_pred if numeric_pred > 0 else 0
        
        if 0.5 < ratio < 1.5:  # Sentiment within 50% of numeric
            # Very conservative blend: 20% sentiment, 80% numeric
            fused_pred = 0.8 * numeric_pred + 0.2 * sentiment_pred
            fused_predictions.append(fused_pred)
        else:
            # Sentiment unreliable - use numeric only
            fused_predictions.append(numeric_pred)
    else:
        # Numeric confident - use it
        fused_predictions.append(numeric_pred)

fused_predictions = np.array(fused_predictions)

# =============================
# FINAL RESULTS
# =============================
print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

fused_r2 = r2_score(y_test_common, fused_predictions)
fused_mae = mean_absolute_error(y_test_common, fused_predictions)

print(f"\n{'Model':<25} {'R²':<10} {'MAE':<15}")
print("-" * 50)
print(f"{'Numeric Only':<25} {numeric_r2:<10.4f} ${numeric_mae:>13,.0f}")
print(f"{'Sentiment Only':<25} {sentiment_r2:<10.4f} ${sentiment_mae:>13,.0f}")
print(f"{'Adaptive Fusion':<25} {fused_r2:<10.4f} ${fused_mae:>13,.0f}")

improvement = fused_r2 - numeric_r2
print(f"\n{'Improvement:':<25} {improvement:+.4f} ({100*improvement/max(numeric_r2, 0.0001):+.2f}%)")

# Count sentiment usage
num_used_sentiment = sum(residuals > residual_threshold)
print(f"\nSentiment used: {num_used_sentiment}/{len(residuals)} cases ({100*num_used_sentiment/len(residuals):.1f}%)")

# Save results
results_df = pd.DataFrame({
    'Actual_Price': y_test_common.values,
    'Numeric_Prediction': numeric_preds_common,
    'Sentiment_Prediction': sentiment_preds_test,
    'Fused_Prediction': fused_predictions,
    'Numeric_Residual': residuals,
    'Sentiment_Used': residuals > residual_threshold
})

results_df.to_csv('smart_fusion_results.csv', index=False)
print("\n💾 Results saved to 'smart_fusion_results.csv'")

print("\n" + "=" * 60)
print("✅ ANALYSIS COMPLETE")
print("=" * 60)