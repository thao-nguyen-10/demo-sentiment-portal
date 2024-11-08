import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from collections import Counter
import seaborn as sns
import numpy as np
import random

# Function to analyze sentiment
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Function to generate a sample dataset
def generate_sample_data(num_rows=100):
    products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
    comments = [
        "Great product, I really love it!",
        "Not what I expected, very disappointed.",
        "It's okay, neither good nor bad.",
        "Fantastic! Will buy again.",
        "Terrible experience, I want a refund.",
        "Average quality, could be better.",
        "Loved it! Highly recommend.",
        "Hated it, will not buy again.",
        "Satisfactory, met my expectations.",
        "Excellent service and product."
    ]
    
    data = {
        'Comment': [random.choice(comments) for _ in range(num_rows)],
        'Product': [random.choice(products) for _ in range(num_rows)],
        'Sales': [random.randint(1, 100) for _ in range(num_rows)],
        'Date': pd.date_range(start='2023-01-01', periods=num_rows, freq='D')
    }

    data_2 = {
        'Product1': [random.choice(products) for _ in range(num_rows)],
        'Product2': [random.choice(products) for _ in range(num_rows)],
        'Sales': [random.randint(1, 100) for _ in range(num_rows)]
    }
    
    return pd.DataFrame(data), pd.DataFrame(data_2)

# Streamlit app
st.title("Sentiment Analysis Dashboard")

# Step 1: Upload Excel file
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Process the uploaded Excel file
    df = pd.read_excel(uploaded_file)
else:
    # Generate sample data if no file is uploaded
    df, df_2 = generate_sample_data()

# Display the data
st.write("Data Preview:")
st.dataframe(df.head())

# Calculate sentiment scores
df['Sentiment Score'] = df['Comment'].apply(analyze_sentiment)

# Step 3: Chart for sentiment score distribution
st.subheader("Sentiment Score Distribution")
plt.figure(figsize=(10, 5))
sns.histplot(df['Sentiment Score'], bins=30, kde=True)
st.pyplot()

# Step 4: Percentage of positive, neutral, and negative comments
sentiment_counts = df['Sentiment Score'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral')).value_counts()
st.subheader("Percentage of Positive, Neutral, and Negative Comments")
st.bar_chart(sentiment_counts)

# Step 5: Sentiment score trends over months
df['Month'] = df['Date'].dt.to_period('M')
monthly_sentiment = df.groupby('Month')['Sentiment Score'].mean().reset_index()

st.subheader("Sentiment Score Trends Over Months")
plt.figure(figsize=(10, 5))
plt.plot(monthly_sentiment['Month'].astype(str), monthly_sentiment['Sentiment Score'], marker='o')
plt.xticks(rotation=45)
plt.title('Monthly Average Sentiment Score')
plt.xlabel('Month')
plt.ylabel('Average Sentiment Score')
st.pyplot()

# Step 6: Top words in positive comments
positive_comments = df[df['Sentiment Score'] > 0]['Comment']
positive_words = ' '.join(positive_comments)
positive_word_counts = Counter(positive_words.split()).most_common(10)

st.subheader("Top Words in Positive Comments")
positive_word_df = pd.DataFrame(positive_word_counts, columns=['Word', 'Count'])
sns.barplot(x='Count', y='Word', data=positive_word_df)
st.pyplot()

# Step 7: Top words in negative comments
negative_comments = df[df['Sentiment Score'] < 0]['Comment']
negative_words = ' '.join(negative_comments)
negative_word_counts = Counter(negative_words.split()).most_common(10)

st.subheader("Top Words in Negative Comments")
negative_word_df = pd.DataFrame(negative_word_counts, columns=['Word', 'Count'])
sns.barplot(x='Count', y='Word', data=negative_word_df)
st.pyplot()

# Step 8: Top 10 selling products and their corresponding sentiment score
top_products = df.groupby('Product').agg({'Sales': 'sum', 'Sentiment Score': 'mean'}).nlargest(10, 'Sales').reset_index()
st.subheader("Top 10 Selling Products and Their Corresponding Sentiment Score")
st.dataframe(top_products)

# Step 9: Top 10 combinations of products and their corresponding sales
product_combinations = df_2.groupby(['Product1', 'Product2']).agg({'Sales': 'sum'}).nlargest(10, 'Sales').reset_index()
st.subheader("Top 10 Combinations of Products and Their Corresponding Sales")
st.dataframe(product_combinations)
