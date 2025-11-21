import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# ---------------------------------------------
# Stopwords
# ---------------------------------------------
STOPWORDS = set([
    "the","and","for","with","but","not","you","all","can","was","are","this","that","have",
    "has","had","she","him","his","her","its","our","your","their","from","into","onto",
    "new","old","near","very","may","now","out","get","see","how","who","why","when",
    "will","would","could","should","about","more","been","being"
])

# ---------------------------------------------
# Extract meaningful words (letters only)
# ---------------------------------------------
def extract_clean_words(text):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', str(text).lower())
    words = [w for w in words if w not in STOPWORDS]
    return words


# ---------------------------------------------
# Build sentiment dictionary for one time segment
# ---------------------------------------------
def build_sentiment_dictionary(descriptions, prices):
    # TF-IDF to find relevant words
    tfidf = TfidfVectorizer(stop_words="english", min_df=2)
    tfidf_matrix = tfidf.fit_transform(descriptions)
    vocab = np.array(tfidf.get_feature_names_out())
    
    sentiment_dict = {}
    
    # Build mapping word → list of prices
    word_price_map = {w: [] for w in vocab}
    
    for doc, price in zip(descriptions, prices):
        words = set(extract_clean_words(doc))
        for w in words:
            if w in word_price_map:
                word_price_map[w].append(price)
    
    # Compute average price and scale 0–1
    avg_prices = []
    for w, plist in word_price_map.items():
        if len(plist) >= 2:
            avg_prices.append(np.mean(plist))
    
    if len(avg_prices) == 0:
        return {}  # no vocabulary
    
    min_p, max_p = min(avg_prices), max(avg_prices)
    rng = max_p - min_p if max_p != min_p else 1
    
    for w, plist in word_price_map.items():
        if len(plist) >= 2:
            avg = np.mean(plist)
            sentiment = (avg - min_p) / rng  # scale 0–1
            sentiment_dict[w] = sentiment
    
    return sentiment_dict


# ---------------------------------------------
# Score sentiment for one row
# ---------------------------------------------
def compute_sentiment(text, dictionary):
    words = extract_clean_words(text)
    score = sum(dictionary.get(w, 0) for w in words)
    return score


# ---------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------
df = pd.read_csv("real_estate_data_chicago.csv")

df = df[['Description', 'Price', 'YearBuilt']].dropna()
df = df[df['Description'].str.strip() != ""]

df['YearBuilt'] = pd.to_numeric(df['YearBuilt'], errors='coerce')
df = df.dropna(subset=['YearBuilt'])

# Ensure numeric prices
def clean_price(p):
    return float(str(p).replace("$","").replace(",","").replace("+",""))

df['Price'] = df['Price'].apply(clean_price)

# ---------------------------------------------
# Split into 20-year bins
# ---------------------------------------------
min_year = int(df['YearBuilt'].min())
max_year = int(df['YearBuilt'].max())

bins = list(range(min_year, max_year + 20, 20))
df['YearBin'] = pd.cut(df['YearBuilt'], bins=bins, right=False)

all_predictions = []
all_actuals = []

print("\n✅ STARTING 20-YEAR SEGMENT ANALYSIS...\n")

for period in df['YearBin'].unique():
    if pd.isna(period):
        continue
    
    segment = df[df['YearBin'] == period].copy()
    if len(segment) < 20:
        continue  # too small
    
    print(f"\n🔹 Processing segment {period}  (rows = {len(segment)})")
    
    descriptions = segment['Description'].tolist()
    prices = segment['Price'].values
    
    # Create sentiment dictionary
    dictionary = build_sentiment_dictionary(descriptions, prices)
    
    if len(dictionary) < 5:
        print("⚠️ Too few informative words — skipping segment.")
        continue
    
    # Sentiment score for each row
    segment['SentimentScore'] = [
        compute_sentiment(desc, dictionary) for desc in descriptions
    ]
    
    X = segment[['SentimentScore']].values
    y = segment['Price'].values
    
    # Train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Random Forest Model
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # Collect predictions
    all_predictions.extend(preds)
    all_actuals.extend(y_test)
    
    # Segment score
    seg_r2 = r2_score(y_test, preds)
    seg_mae = mean_absolute_error(y_test, preds)
    print(f"   ✅ Segment R²: {seg_r2:.4f}, MAE: {seg_mae:,.0f}")


# ---------------------------------------------
# FINAL GLOBAL RESULTS
# ---------------------------------------------
all_predictions = np.array(all_predictions)
all_actuals = np.array(all_actuals)

if len(all_predictions) > 0:
    final_r2 = r2_score(all_actuals, all_predictions)
    final_mae = mean_absolute_error(all_actuals, all_predictions)

    print("\n✅ FINAL SENTIMENT-ONLY MODEL RESULTS")
    print("=======================================")
    print(f"R²:  {final_r2}")
    print(f"MAE: {final_mae:,.2f}")
    
    # Save predictions
    df_out = pd.DataFrame({
        'ActualPrice': all_actuals,
        'PredictedPrice': all_predictions,
        'Error': np.abs(all_predictions - all_actuals)
    })
    df_out.to_excel("sentiment_predictions_segmented.xlsx", index=False)
    print("\n✅ Saved: sentiment_predictions_segmented.xlsx")

else:
    print("\n❌ No valid segments — model could not run.")
