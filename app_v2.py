import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta, datetime
import random
from textblob import TextBlob
from collections import Counter

# Function to analyze sentiment
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Function to generate a sample dataset
def generate_sample_data(num_rows=100):
    products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
    customers = ['Customer A', 'Customer B', 'Customer C']
    shops = ['Shop 1', 'Shop 2', 'Shop 3', 'Shop 4', 'Shop 5']
    customer_type = ['Diamond','Gold','Silver','Newbie']
    comments_product = [
        "It is made by 100% cotton fabric, I really love it!",
        "The size is too big, very disappointed.",
        "Design is okay, fair material, neither good nor bad.",
        "Fantastic design and material! Will buy again.",
        "Terrible material, also rip, I want a refund.",
        "Average design, easy to be creased, could be better.",
        "Loved the design and material! Highly recommend.",
        "Hated because the actual item is not like it in the image, will not buy again.",
        "Good material, basic design, met my expectations.",
        "Excellent design but a bit expensive."
    ]
    comments_service = [
        "Five star service, I really love it!",
        "Too long delivery, very disappointed.",
        "It's okay, neither good nor bad.",
        "Fantastic packaging! Will buy again.",
        "Terrible experience, I want a refund.",
        "Fair enough, could be better.",
        "Loved it! Highly recommend.",
        "Hated it, will not buy again.",
        "Satisfactory, met my expectations.",
        "Excellent service and packaging."
    ]
    
    data = {
        'Product Comment': [random.choice(comments_product) for _ in range(num_rows)],
        'Service Comment': [random.choice(comments_service) for _ in range(num_rows)],
        'Product': [random.choice(products) for _ in range(num_rows)],
        'Customer': [random.choice(customers) for _ in range(num_rows)],
        'Customer Type': [random.choice(customer_type) for _ in range(num_rows)],
        'Shop': [random.choice(shops) for _ in range(num_rows)],
        'Sales': [random.randint(1, 100) for _ in range(num_rows)],
        'Date': pd.date_range(start='2023-01-01', periods=num_rows, freq='D')
    }

    data['Comment'] = [
    f'{product_comment} {service_comment}'
    for product_comment, service_comment in zip(data['Product Comment'], data['Service Comment'])
    ]
    return pd.DataFrame(data)

# Function to load data
def load_data(uploaded_file):
    if uploaded_file is not None:
        # Assuming the uploaded file is a CSV
        data = pd.read_csv(uploaded_file)
        return data
    return None

# Streamlit App
# Page configuration
st.set_page_config(page_title="Multi-Page App", layout="centered")
st.title("Business Analytics Dashboard")

def page1_overview():
    st.title("Overview sentiment report")
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

    if uploaded_file is not None:
        # Process the uploaded Excel file
        df = pd.read_excel(uploaded_file)
    else:
        # Generate sample data if no file is uploaded
        df = generate_sample_data()
        
    st.session_state.data = df

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

def page2_product():
    st.title("Product Sentiment Report")

    if 'data' in st.session_state:
        df = st.session_state.data
        
        # Calculate sentiment scores
        df['Product Sentiment Score'] = df['Product Comment'].apply(analyze_sentiment)

        # Step 3: Chart for sentiment score distribution
        st.subheader("Sentiment Score Distribution")
        plt.figure(figsize=(10, 5))
        sns.histplot(df['Product Sentiment Score'], bins=30, kde=True)
        st.pyplot()

        # Step 4: Percentage of positive, neutral, and negative comments
        sentiment_counts = df['Product Sentiment Score'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral')).value_counts()
        st.subheader("Percentage of Positive, Neutral, and Negative Comments")
        st.bar_chart(sentiment_counts)

        # Step 5: Sentiment score trends over months
        df['Month'] = df['Date'].dt.to_period('M')
        monthly_sentiment = df.groupby('Month')['Product Sentiment Score'].mean().reset_index()

        st.subheader("Sentiment Score Trends Over Months")
        plt.figure(figsize=(10, 5))
        plt.plot(monthly_sentiment['Month'].astype(str), monthly_sentiment['Product Sentiment Score'], marker='o')
        plt.xticks(rotation=45)
        plt.title('Monthly Average Sentiment Score')
        plt.xlabel('Month')
        plt.ylabel('Average Sentiment Score')
        st.pyplot()

        # Step 6: Top words in positive comments
        positive_comments = df[df['Product Sentiment Score'] > 0]['Product Comment']
        positive_words = ' '.join(positive_comments)
        positive_word_counts = Counter(positive_words.split()).most_common(10)

        st.subheader("Top Words in Positive Comments")
        positive_word_df = pd.DataFrame(positive_word_counts, columns=['Word', 'Count'])
        sns.barplot(x='Count', y='Word', data=positive_word_df)
        st.pyplot()

        # Step 7: Top words in negative comments
        negative_comments = df[df['Product Sentiment Score'] < 0]['Product Comment']
        negative_words = ' '.join(negative_comments)
        negative_word_counts = Counter(negative_words.split()).most_common(10)

        st.subheader("Top Words in Negative Comments")
        negative_word_df = pd.DataFrame(negative_word_counts, columns=['Word', 'Count'])
        sns.barplot(x='Count', y='Word', data=negative_word_df)
        st.pyplot()

        # Step 8: Top 10 selling products and their corresponding sentiment score
        top_products = df.groupby('Product').agg({'Sales': 'sum', 'Product Sentiment Score': 'mean'}).nlargest(10, 'Sales').reset_index()
        st.subheader("Top 10 Selling Products and Their Corresponding Sentiment Score")
        st.dataframe(top_products)

        # Step 9: Top 10 products with lowest sentiment score and their corresponding sales
        bottom_sentiment = df.groupby('Product').agg({'Sales': 'sum', 'Product Sentiment Score': 'mean'}).nsmallest(10, 'Product Sentiment Score').reset_index()
        st.subheader("10 Products with Lowest Sentiment Score and Their Corresponding Sale")
        st.dataframe(bottom_sentiment)

def page3_service():
    st.title("Service Quality Monitoring")
    if 'data' in st.session_state:
        df = st.session_state.data
    
        # Calculate sentiment scores
        df['Service Sentiment Score'] = df['Service Comment'].apply(analyze_sentiment)

        # Step 3: Chart for sentiment score distribution
        st.subheader("Sentiment Score Distribution")
        plt.figure(figsize=(10, 5))
        sns.histplot(df['Service Sentiment Score'], bins=30, kde=True)
        st.pyplot()

        # Step 4: Percentage of positive, neutral, and negative comments
        sentiment_counts = df['Service Sentiment Score'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral')).value_counts()
        st.subheader("Percentage of Positive, Neutral, and Negative Comments")
        st.bar_chart(sentiment_counts)

        # Step 5: Sentiment score trends over months
        df['Month'] = df['Date'].dt.to_period('M')
        monthly_sentiment = df.groupby('Month')['Service Sentiment Score'].mean().reset_index()

        st.subheader("Sentiment Score Trends Over Months")
        plt.figure(figsize=(10, 5))
        plt.plot(monthly_sentiment['Month'].astype(str), monthly_sentiment['Service Sentiment Score'], marker='o')
        plt.xticks(rotation=45)
        plt.title('Monthly Average Sentiment Score')
        plt.xlabel('Month')
        plt.ylabel('Average Sentiment Score')
        st.pyplot()

        # Step 6: Top words in positive comments
        positive_comments = df[df['Service Sentiment Score'] > 0]['Service Comment']
        positive_words = ' '.join(positive_comments)
        positive_word_counts = Counter(positive_words.split()).most_common(10)

        st.subheader("Top Words in Positive Comments")
        positive_word_df = pd.DataFrame(positive_word_counts, columns=['Word', 'Count'])
        sns.barplot(x='Count', y='Word', data=positive_word_df)
        st.pyplot()

        # Step 7: Top words in negative comments
        negative_comments = df[df['Service Sentiment Score'] < 0]['Service Comment']
        negative_words = ' '.join(negative_comments)
        negative_word_counts = Counter(negative_words.split()).most_common(10)

        st.subheader("Top Words in Negative Comments")
        negative_word_df = pd.DataFrame(negative_word_counts, columns=['Word', 'Count'])
        sns.barplot(x='Count', y='Word', data=negative_word_df)
        st.pyplot()

        # Customer segmentation for stacked column chart
        # Calculate average and 75th percentile of service sentiment score
        avg_positive_reviews = df['Service Sentiment Score'].mean()
        p75_positive_reviews = df['Service Sentiment Score'].quantile(0.75)

        # Categorize customers
        df['Customer Segment'] = pd.cut(
            df['Service Sentiment Score'],
            bins=[-1, avg_positive_reviews, p75_positive_reviews, df['Service Sentiment Score'].max()],
            labels=['Less than Average', 'Average to 75th Percentile', 'More than 75th Percentile']
        )

        # Count customers in each category and membership level
        customer_counts = df.groupby(['Customer Type', 'Customer Segment']).size().unstack(fill_value=0)

        # Stacked column chart
        st.subheader("Customer Segmentation by Positive Reviews and Membership Level")
        customer_counts.plot(kind='bar', stacked=True)
        plt.title('Customer Segmentation')
        plt.xlabel('Customer Segment')
        plt.ylabel('Number of Customers')
        st.pyplot()

def page4_competitor():
    st.title("Competitor Analysis")

    if 'data' in st.session_state:
        df = st.session_state.data

        # Calculate sentiment scores
        df['Sentiment Score'] = df['Comment'].apply(analyze_sentiment)
        df['Product Sentiment Score'] = df['Product Comment'].apply(analyze_sentiment)
        df['Service Sentiment Score'] = df['Service Comment'].apply(analyze_sentiment)
        df['Month'] = df['Date'].dt.to_period('M')
        shop_df = df[df['Shop'] == 'Shop 1']
        
        # Step 1: Filter sentiment for the specific shop
        monthly_sentiment = df.groupby('Month')['Sentiment Score'].mean().reset_index()
        shop_monthly_sentiment = shop_df.groupby('Month')['Sentiment Score'].mean().reset_index()
        shop_monthly_sentiment.rename(columns={'Sentiment Score':'Shop Sentiment Score'}, inplace=True)
        plot_df = monthly_sentiment.merge(shop_monthly_sentiment, on='Month', how='left')

        # Line chart for sentiment score of shop vs all shops
        st.subheader("Sentiment Score of Shop vs All Shops")
        plt.figure(figsize=(10, 5))
        plt.plot(plot_df['Month'].astype(str), plot_df['Sentiment Score'], label='Global Sentiment Score', marker='o')
        plt.plot(plot_df['Month'].astype(str), plot_df['Shop Sentiment Score'], label='Shop Sentiment Score', marker='o')
        plt.xticks(rotation=45)
        plt.title('Monthly Average Sentiment Score')
        plt.xlabel('Month')
        plt.ylabel('Average Sentiment Score')
        st.pyplot()

        # Step 2: Filter product sentiment for the specific shop
        monthly_product_sentiment = df.groupby('Month')['Product Sentiment Score'].mean().reset_index()
        shop_monthly_product_sentiment = shop_df.groupby('Month')['Product Sentiment Score'].mean().reset_index()
        shop_monthly_product_sentiment.rename(columns={'Product Sentiment Score':'Shop Product Sentiment Score'}, inplace=True)
        plot_df_2 = monthly_product_sentiment.merge(shop_monthly_product_sentiment, on='Month', how='left')

        # Line chart for product sentiment score of shop vs all shops
        st.subheader("Product Sentiment Score of Shop vs All Shops")
        plt.figure(figsize=(10, 5))
        plt.plot(plot_df_2['Month'].astype(str), plot_df_2['Product Sentiment Score'], label='Global Product Sentiment Score', marker='o')
        plt.plot(plot_df_2['Month'].astype(str), plot_df_2['Shop Product Sentiment Score'], label='Shop Product Sentiment Score', marker='o')
        plt.xticks(rotation=45)
        plt.title('Monthly Average Product Sentiment Score')
        plt.xlabel('Month')
        plt.ylabel('Average Product Sentiment Score')
        st.pyplot()

        # Step 3: Filter service sentiment for the specific shop
        monthly_service_sentiment = df.groupby('Month')['Service Sentiment Score'].mean().reset_index()
        shop_monthly_service_sentiment = shop_df.groupby('Month')['Service Sentiment Score'].mean().reset_index()
        shop_monthly_service_sentiment.rename(columns={'Service Sentiment Score':'Shop Service Sentiment Score'}, inplace=True)
        plot_df_3 = monthly_service_sentiment.merge(shop_monthly_service_sentiment, on='Month', how='left')

        # Line chart for product sentiment score of shop vs all shops
        st.subheader("Service Sentiment Score of Shop vs All Shops")
        plt.figure(figsize=(10, 5))
        plt.plot(plot_df_3['Month'].astype(str), plot_df_3['Service Sentiment Score'], label='Global Service Sentiment Score', marker='o')
        plt.plot(plot_df_3['Month'].astype(str), plot_df_3['Shop Service Sentiment Score'], label='Shop Service Sentiment Score', marker='o')
        plt.xticks(rotation=45)
        plt.title('Monthly Average Service Sentiment Score')
        plt.xlabel('Month')
        plt.ylabel('Average Service Sentiment Score')
        st.pyplot()

        # Step 4: Calculate the gap in sentiment scores
        product_sentiment = df.groupby('Product')['Product Sentiment Score'].mean().reset_index()
        product_sentiment.rename(columns={'Product Sentiment Score':'Global Product Sentiment Score'}, inplace=True)
        shop_product_sentiment = shop_df.groupby('Product')['Product Sentiment Score'].mean().reset_index()
        shop_product_sentiment = shop_product_sentiment.merge(product_sentiment, on='Product', how='left')
        shop_product_sentiment['Sentiment Gap'] = shop_product_sentiment['Product Sentiment Score'] - shop_product_sentiment['Global Product Sentiment Score']

        # Get the top 10 products with the largest gap in sentiment score
        negative_gap = shop_product_sentiment[shop_product_sentiment['Sentiment Gap'] < 0]
        top_products_gap = negative_gap.nsmallest(10, 'Sentiment Gap')

        # Display the table
        st.subheader("Top 10 Products with the Largest Gap in Sentiment Score")
        st.dataframe(top_products_gap)

# Main app logic
if __name__ == "__main__":
    pages = {
        "Collect data and Overview": page1_overview,
        "Product Sentiment": page2_product,
        "Service Sentiment": page3_service,
        "Competitor Analysis": page4_competitor
    }
    
    selected_page = st.sidebar.selectbox("Select a page", list(pages.keys()))
    pages[selected_page]()
