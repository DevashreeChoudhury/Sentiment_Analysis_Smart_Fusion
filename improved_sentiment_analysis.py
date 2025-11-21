import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

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

# ===================== MAIN PIPELINE =====================

df = pd.read_csv("real_estate_data_chicago.csv")
df = df[['Description', 'Price', 'YearBuilt', 'Sqft', 'Bedrooms']].dropna()
df = df[df['Description'].str.strip() != ""]

df['YearBuilt'] = pd.to_numeric(df['YearBuilt'], errors='coerce')
df['Sqft'] = pd.to_numeric(df['Sqft'], errors='coerce')
df['Bedrooms'] = pd.to_numeric(df['Bedrooms'], errors='coerce')
df = df.dropna(subset=['YearBuilt', 'Sqft', 'Bedrooms'])

def clean_price(p):
    return float(str(p).replace("$","").replace(",","").replace("+",""))

df['Price'] = df['Price'].apply(clean_price)

# Year bins
min_year = int(df['YearBuilt'].min()); max_year = int(df['YearBuilt'].max())
year_bins = list(range(min_year, max_year + 20, 20))
df['YearBin'] = pd.cut(df['YearBuilt'], bins=year_bins, right=False)

# Sqft bins
sqft_bins = [0, 900, 1500, 2200, 3200, 5000, 10000]
df['SqftBin'] = pd.cut(df['Sqft'], bins=sqft_bins, right=False)

# Bedroom bins
bed_bins = [0, 1, 2, 3, 4, 20]
df['BedBin'] = pd.cut(df['Bedrooms'], bins=bed_bins, right=False)

print("\n🚀 STARTING SENTIMENT ANALYSIS (Year × Sqft × Bedroom)...\n")

all_predictions = []
all_actuals = []

for yperiod in df['YearBin'].unique():
    if pd.isna(yperiod): continue
    df_y = df[df['YearBin'] == yperiod]

    for speriod in df_y['SqftBin'].unique():
        if pd.isna(speriod): continue
        df_ys = df_y[df_y['SqftBin'] == speriod]

        for bperiod in df_ys['BedBin'].unique():
            if pd.isna(bperiod): continue
            segment = df_ys[df_ys['BedBin'] == bperiod].copy()

            if len(segment) < 20:
                continue

            print(f"\n🔹 Segment {yperiod} | {speriod} | {bperiod}  (rows={len(segment)})")

            descriptions = segment['Description'].tolist()
            prices = segment['Price'].values

            dictionary = build_sentiment_dictionary(descriptions, prices)
            if len(dictionary) < 5:
                print("⚠️ Too few words — skipping")
                continue

            segment['SentimentScore'] = [
                compute_sentiment(desc, dictionary) for desc in descriptions
            ]

            X = segment[['SentimentScore']].values
            y = segment['Price'].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = RandomForestRegressor(
                n_estimators=450, max_depth=None, random_state=42, n_jobs=-1
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            all_predictions.extend(preds)
            all_actuals.extend(y_test)

            print(f"   ✔ R² = {r2_score(y_test, preds):.4f} | MAE = {mean_absolute_error(y_test, preds):,.0f}")

# ===================== FINAL RESULTS =====================

# ---------------------------------------------
# FINAL GLOBAL RESULTS
# ---------------------------------------------
all_predictions = np.array(all_predictions)
all_actuals = np.array(all_actuals)

if len(all_predictions) > 0:
    final_r2 = r2_score(all_actuals, all_predictions)
    final_mae = mean_absolute_error(all_actuals, all_predictions)

    print("\n🏁 FINAL SENTIMENT-ONLY MODEL RESULTS")
    print("=======================================")
    print(f"R²:  {final_r2}")
    print(f"MAE: {final_mae:,.2f}")

    df_out = pd.DataFrame({
        "ActualPrice": all_actuals,
        "PredictedPrice": all_predictions,
        "Error": np.abs(all_predictions - all_actuals)
    })
    df_out.to_excel("sentiment_predictions_segmented.xlsx", index=False)
    print("📁 Saved: sentiment_predictions_segmented.xlsx")
else:
    print("\n❌ No valid segments — model could not run.")
