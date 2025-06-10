import pandas as pd
import numpy as np
import re
from textblob import TextBlob
import string
import os
import glob

# Find the CSV file automatically
csv_files = glob.glob('*.csv') + glob.glob('**/*.csv', recursive=True)
economic_files = [f for f in csv_files if 'economic' in f.lower()]

if not economic_files:
    print("Available CSV files:")
    for f in csv_files[:10]:  # Show first 10 CSV files
        print(f"  {f}")
    filename = input("Enter the exact filename of your CSV: ")
else:
    filename = economic_files[0]
    print(f"Found CSV file: {filename}")

# Read the CSV file
try:
    df = pd.read_csv(filename)
    print(f"Successfully loaded: {filename}")
except Exception as e:
    print(f"Error loading {filename}: {e}")
    # Try different encodings
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(filename, encoding=encoding)
            print(f"Successfully loaded with {encoding} encoding")
            break
        except:
            continue
    else:
        raise Exception("Could not read the CSV file with any encoding")

print(f"Original data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Clean the data
def clean_data(df):
    # Remove rows where all values are NaN or empty
    df = df.dropna(how='all')
    
    # Replace placeholder values with NaN
    placeholder_patterns = ['###', '####', '#######', '########', 'NULL', 'null', 'N/A', 'n/a', '']
    for pattern in placeholder_patterns:
        df = df.replace(pattern, np.nan)
        df = df.replace(f'^{re.escape(pattern)}+$', np.nan, regex=True)
    
    # Clean whitespace
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace('nan', np.nan)
    
    # Remove duplicate rows
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed_duplicates = initial_rows - len(df)
    print(f"Removed {removed_duplicates} duplicate rows")
    
    # Clean URLs - ensure they start with http
    url_columns = [col for col in df.columns if 'link' in col.lower() or 'url' in col.lower()]
    for col in url_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if pd.isna(x) or str(x).startswith(('http://', 'https://')) else f'https://{x}' if str(x) != 'nan' else np.nan)
    
    return df

# Perform tagging
def perform_tagging(df):
    # Identify text columns for tagging
    text_columns = []
    for col in df.columns:
        if df[col].dtype == 'object' and col.lower() not in ['link', 'url', 'source']:
            # Check if column contains text (not just URLs or short codes)
            sample_values = df[col].dropna().head(10)
            if any(len(str(val)) > 20 for val in sample_values):
                text_columns.append(col)
    
    print(f"Text columns for tagging: {text_columns}")
    
    # Define economic topics and keywords
    economic_topics = {
        'inflation': ['inflation', 'price rise', 'cost increase', 'cpi', 'consumer price'],
        'employment': ['job', 'employment', 'unemployment', 'hiring', 'layoff', 'workforce'],
        'gdp': ['gdp', 'growth', 'economic growth', 'expansion', 'recession'],
        'monetary_policy': ['interest rate', 'fed', 'central bank', 'monetary policy', 'quantitative easing'],
        'trade': ['trade', 'import', 'export', 'tariff', 'trade war', 'commerce'],
        'stock_market': ['stock', 'market', 'dow', 'nasdaq', 's&p', 'equity', 'shares'],
        'banking': ['bank', 'credit', 'loan', 'mortgage', 'financial institution'],
        'currency': ['dollar', 'euro', 'currency', 'exchange rate', 'forex'],
        'corporate': ['company', 'corporate', 'business', 'earnings', 'profit', 'revenue'],
        'real_estate': ['housing', 'real estate', 'property', 'mortgage', 'home prices'],
        'energy': ['oil', 'gas', 'energy', 'petroleum', 'crude', 'renewable'],
        'technology': ['tech', 'technology', 'digital', 'ai', 'artificial intelligence', 'crypto']
    }
    
    # Sentiment analysis
    def get_sentiment(text):
        if pd.isna(text) or str(text) == 'nan':
            return 'neutral'
        try:
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            if polarity > 0.1:
                return 'positive'
            elif polarity < -0.1:
                return 'negative'
            else:
                return 'neutral'
        except:
            return 'neutral'
    
    # Topic tagging
    def tag_topics(text):
        if pd.isna(text) or str(text) == 'nan':
            return []
        
        text_lower = str(text).lower()
        found_topics = []
        
        for topic, keywords in economic_topics.items():
            if any(keyword in text_lower for keyword in keywords):
                found_topics.append(topic)
        
        return found_topics if found_topics else ['general_economic']
    
    # Country/Region detection
    def detect_region(text):
        if pd.isna(text) or str(text) == 'nan':
            return 'unknown'
        
        text_lower = str(text).lower()
        regions = {
            'usa': ['us ', 'usa', 'united states', 'america', 'american'],
            'europe': ['europe', 'european', 'eu ', 'eurozone'],
            'uk': ['uk', 'britain', 'british', 'england', 'london'],
            'china': ['china', 'chinese', 'beijing'],
            'india': ['india', 'indian', 'mumbai', 'delhi'],
            'japan': ['japan', 'japanese', 'tokyo'],
            'global': ['global', 'world', 'international', 'worldwide']
        }
        
        for region, keywords in regions.items():
            if any(keyword in text_lower for keyword in keywords):
                return region
        
        return 'other'
    
    # Apply tagging to main text column
    main_text_col = None
    for col in text_columns:
        if 'title' in col.lower() or 'description' in col.lower() or len(text_columns) == 1:
            main_text_col = col
            break
    
    if main_text_col is None and text_columns:
        main_text_col = text_columns[0]
    
    if main_text_col:
        print(f"Applying tags based on column: {main_text_col}")
        
        # Add sentiment
        df['sentiment'] = df[main_text_col].apply(get_sentiment)
        
        # Add topics
        df['topics'] = df[main_text_col].apply(lambda x: ', '.join(tag_topics(x)))
        
        # Add region
        df['region'] = df[main_text_col].apply(detect_region)
        
        # Add text length category
        df['content_length'] = df[main_text_col].apply(lambda x: 'short' if pd.isna(x) or len(str(x)) < 100 else 'medium' if len(str(x)) < 300 else 'long')
        
        # Add urgency/importance (based on certain keywords)
        urgent_keywords = ['breaking', 'urgent', 'crisis', 'crash', 'emergency', 'immediate']
        df['urgency'] = df[main_text_col].apply(lambda x: 'high' if pd.isna(x) or str(x) == 'nan' else 'high' if any(keyword in str(x).lower() for keyword in urgent_keywords) else 'normal')
    
    return df

# Execute cleaning and tagging
print("Starting data cleaning...")
df_cleaned = clean_data(df)

print(f"After cleaning shape: {df_cleaned.shape}")
print(f"Missing values per column:")
print(df_cleaned.isnull().sum())

print("\nStarting tagging...")
df_final = perform_tagging(df_cleaned)

print(f"Final data shape: {df_final.shape}")
print(f"New columns added: {[col for col in df_final.columns if col not in df.columns]}")

# Save the cleaned and tagged data
output_filename = 'economic_news_cleaned_tagged.csv'
df_final.to_csv(output_filename, index=False)
print(f"\nCleaned and tagged data saved to: {output_filename}")

# Display sample of cleaned data
print("\nSample of cleaned and tagged data:")
print(df_final.head())

# Data quality report
print("\n" + "="*50)
print("DATA QUALITY REPORT")
print("="*50)
print(f"Original rows: {df.shape[0]}")
print(f"Final rows: {df_final.shape[0]}")
print(f"Rows removed: {df.shape[0] - df_final.shape[0]}")
print(f"Completion rate: {((df_final.notna().sum().sum()) / (df_final.shape[0] * df_final.shape[1]) * 100):.1f}%")

if 'sentiment' in df_final.columns:
    print(f"\nSentiment distribution:")
    print(df_final['sentiment'].value_counts())

if 'region' in df_final.columns:
    print(f"\nRegion distribution:")
    print(df_final['region'].value_counts())

if 'topics' in df_final.columns:
    print(f"\nTop topics:")
    all_topics = []
    for topics_str in df_final['topics'].dropna():
        all_topics.extend([t.strip() for t in str(topics_str).split(',')])
    from collections import Counter
    topic_counts = Counter(all_topics)
    for topic, count in topic_counts.most_common(10):
        print(f"  {topic}: {count}")

print(f"\nFinal assessment: {'CLEAN' if df_final.isnull().sum().sum() < (df_final.shape[0] * 0.1) else 'MOSTLY CLEAN'}")
print("="*50)