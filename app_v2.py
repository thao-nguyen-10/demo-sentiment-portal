import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta, datetime

# Sample Data Generation
def generate_sample_data():
    np.random.seed(42)
    weeks = pd.date_range(datetime.now() - timedelta(weeks=8), periods=8).date
    products = [f'Product {i}' for i in range(1, 11)]
    services = ['Service A', 'Service B']

    # Sentiment scores
    product_sentiment = pd.DataFrame({
        'Date': np.tile(weeks, len(products)),
        'Product': np.repeat(products, len(weeks)),
        'Sentiment Score': np.random.rand(len(weeks) * len(products)) * 2 - 1  # Between -1 and 1
    })

    # Feedback counts
    feedback_counts = pd.DataFrame({
        'Date': np.tile(weeks, len(products)),
        'Product': np.repeat(products, len(weeks)),
        'Positive': np.random.randint(10, 100, size=len(weeks) * len(products)),
        'Neutral': np.random.randint(5, 50, size=len(weeks) * len(products)),
        'Negative': np.random.randint(1, 10, size=len(weeks) * len(products)),
    })

    # Service sentiment
    service_sentiment = pd.DataFrame({
        'Date': np.tile(weeks, len(services)),
        'Service': np.repeat(services, len(weeks)),
        'Sentiment Score': np.random.rand(len(weeks) * len(services)) * 2 - 1
    })

    return product_sentiment, feedback_counts, service_sentiment

# Load sample data
product_sentiment, feedback_counts, service_sentiment = generate_sample_data()

# Streamlit App
st.title("Business Analytics Dashboard")

# Sidebar for navigation
pages = {
    "Product Sentiment Analysis": "product",
    "Service Quality Monitoring": "service",
    "Competitor Analysis": "competitor"
}

page = st.sidebar.selectbox("Select a page", list(pages.keys()))

if pages[page] == "product":
    st.header("Product Sentiment Analysis")

    st.subheader("Sentiment Score Trends Over Weeks")
    # Line chart for sentiment score of all products    
    sentiment_score_chart = product_sentiment.pivot(index='Date', columns='Product', values='Sentiment Score')
    st.line_chart(sentiment_score_chart)

    
    # Column chart for positive, neutral, negative feedbacks
    st.subheader("Feedback Summary Over Weeks")
    feedback_summary = feedback_counts.groupby('Date').sum().reset_index()
    feedback_summary.set_index('Date', inplace=True)
    feedback_summary.plot(kind='bar', stacked=True)
    plt.title('Feedback Summary Over 8 Weeks')
    plt.xlabel('Date')
    plt.ylabel('Number of Feedbacks')
    st.pyplot()

    # Top 10 words in positive and negative reviews (dummy data)
    words_positive = pd.Series(np.random.choice(['great', 'excellent', 'good', 'love', 'amazing'], 100)).value_counts().head(10)
    words_negative = pd.Series(np.random.choice(['bad', 'terrible', 'awful', 'hate', 'poor'], 100)).value_counts().head(10)

    # Bar charts for top words
    st.subheader("Top 10 Words in Positive Reviews about Product")
    st.bar_chart(words_positive)

    st.subheader("Top 10 Words in Negative Reviews about Product")
    st.bar_chart(words_negative)

    # Growth rate tables (dummy data)
    growth_positive = feedback_counts.groupby('Product')['Positive'].mean().nlargest(10)
    growth_negative = feedback_counts.groupby('Product')['Negative'].mean().nlargest(10)
    st.write("Top 10 Products by Positive Growth Rate")
    st.dataframe(growth_positive)

    st.write("Top 10 Products by Negative Growth Rate")
    st.dataframe(growth_negative)

elif pages[page] == "service":
    st.header("Service Quality Monitoring")

    # Line chart for sentiment score of service
    st.subheader("Sentiment Score Trends Over Weeks")
    service_sentiment_chart = service_sentiment.pivot(index='Date', columns='Service', values='Sentiment Score')
    st.line_chart(service_sentiment_chart)

    # Column chart for positive, neutral, negative feedbacks about service
    st.subheader("Feedback Summary Over Weeks")
    service_feedback_summary = feedback_counts.groupby('Date').sum().reset_index()
    service_feedback_summary.set_index('Date', inplace=True)
    service_feedback_summary[['Positive', 'Neutral', 'Negative']].plot(kind='bar', stacked=True)
    plt.title('Service Feedback Summary Over 8 Weeks')
    plt.xlabel('Date')
    plt.ylabel('Number of Feedbacks')
    st.pyplot()

    # Top 10 words in positive reviews (dummy data)
    words_positive_service = pd.Series(np.random.choice(['excellent', 'helpful', 'friendly', 'quick', 'responsive'], 100)).value_counts().head(10)
    st.subheader("Top 10 Words in Positive Reviews about Service")
    st.bar_chart(words_positive_service)

    # Top 10 words in negative reviews (dummy data)
    words_negative_service = pd.Series(np.random.choice(['slow', 'rude', 'unhelpful', 'poor', 'bad'], 100)).value_counts().head(10)
    st.subheader("Top 10 Words in Negative Reviews about Service")
    st.bar_chart(words_negative_service)

    # Customer segmentation for stacked column chart
    # Generating dummy customer data
    customer_data = pd.DataFrame({
        'Customer ID': range(1, 101),
        'Positive Reviews': np.random.randint(1, 100, size=100),
        'Membership Level': np.random.choice(['Diamond', 'Gold', 'Silver'], size=100)
    })

    # Calculate average and 75th percentile of positive reviews
    avg_positive_reviews = customer_data['Positive Reviews'].mean()
    p75_positive_reviews = customer_data['Positive Reviews'].quantile(0.75)

    # Categorize customers
    customer_data['Customer Type'] = pd.cut(
        customer_data['Positive Reviews'],
        bins=[-1, avg_positive_reviews, p75_positive_reviews, customer_data['Positive Reviews'].max()],
        labels=['Less than Average', 'Average to 75th Percentile', 'More than 75th Percentile']
    )

    # Count customers in each category and membership level
    customer_counts = customer_data.groupby(['Customer Type', 'Membership Level']).size().unstack(fill_value=0)

    # Stacked column chart
    st.subheader("Customer Segmentation by Positive Reviews and Membership Level")
    customer_counts.plot(kind='bar', stacked=True)
    plt.title('Customer Segmentation')
    plt.xlabel('Customer Type')
    plt.ylabel('Number of Customers')
    st.pyplot()

elif pages[page] == "competitor":
    st.header("Competitor Analysis")

    # Sample data for competitor analysis
    # Generating sentiment scores for a specific shop and all shops
    shop_name = "Shop A"
    all_shops = ["Shop A", "Shop B", "Shop C", "Shop D", "Shop E"]
    weeks = pd.date_range(datetime.now() - timedelta(weeks=8), periods=8)

    # Sentiment scores for each shop
    shop_sentiment = pd.DataFrame({
        'Date': np.tile(weeks, len(all_shops)),
        'Shop': np.repeat(all_shops, len(weeks)),
        'Sentiment Score': np.random.rand(len(weeks) * len(all_shops)) * 2 - 1  # Between -1 and 1
    })

    # Filter sentiment for the specific shop
    shop_sentiment_filtered = shop_sentiment[shop_sentiment['Shop'] == shop_name]
    all_shops_sentiment = shop_sentiment.pivot(index='Date', columns='Shop', values='Sentiment Score')

    # Line chart for sentiment score of shop vs all shops
    st.subheader("Sentiment Score of Shop vs All Shops")
    st.line_chart(all_shops_sentiment)

    # Column chart for feedback comparison
    feedback_counts_competitor = pd.DataFrame({
        'Date': np.tile(weeks, len(all_shops)),
        'Shop': np.repeat(all_shops, len(weeks)),
        'Positive': np.random.randint(10, 100, size=len(weeks) * len(all_shops)),
        'Neutral': np.random.randint(5, 50, size=len(weeks) * len(all_shops)),
        'Negative': np.random.randint(1, 10, size=len(weeks) * len(all_shops)),
    })

    feedback_summary_competitor = feedback_counts_competitor.groupby('Date').sum().reset_index()
    feedback_summary_competitor.set_index('Date', inplace=True)
    feedback_summary_competitor[['Positive', 'Neutral', 'Negative']].plot(kind='bar', stacked=True)
    plt.title('Feedback Summary for Shops Over 8 Weeks')
    plt.xlabel('Date')
    plt.ylabel('Number of Feedbacks')
    st.pyplot()

    # Top words in positive reviews (dummy data for shop and all shops)
    words_positive_shop = pd.Series(np.random.choice(['great', 'excellent', 'good', 'love', 'amazing'], 100)).value_counts().head(10)
    words_positive_all = pd.Series(np.random.choice(['great', 'awesome', 'fantastic', 'superb', 'excellent'], 100)).value_counts().head(10)

    # Bar chart for top words in positive reviews
    st.subheader("Top 10 Words in Positive Reviews")
    fig, ax = plt.subplots()
    words_positive_shop.plot(kind='bar', color='blue', ax=ax, alpha=0.6, label=shop_name)
    words_positive_all.plot(kind='bar', color='orange', ax=ax, alpha=0.6, label='All Shops')
    plt.title('Top 10 Words in Positive Reviews')
    plt.ylabel('Frequency')
    plt.xlabel('Words')
    plt.legend()
    st.pyplot()

    # Top words in negative reviews (dummy data for shop and all shops)
    words_negative_shop = pd.Series(np.random.choice(['bad', 'terrible', 'awful', 'hate', 'poor'], 100)).value_counts().head(10)
    words_negative_all = pd.Series(np.random.choice(['horrible', 'terrible', 'bad', 'worst', 'unacceptable'], 100)).value_counts().head(10)

    # Bar chart for top words in negative reviews
    st.subheader("Top 10 Words in Negative Reviews")
    fig, ax = plt.subplots()
    words_negative_shop.plot(kind='bar', color='red', ax=ax, alpha=0.6, label=shop_name)
    words_negative_all.plot(kind='bar', color='gray', ax=ax, alpha=0.6, label='All Shops')
    plt.title('Top 10 Words in Negative Reviews')
    plt.ylabel('Frequency')
    plt.xlabel('Words')
    plt.legend()
    st.pyplot()

    # Table for top 10 products with the largest gap in sentiment score
        # Continuing from the previous code...

    # Generating dummy data for product sentiment scores
    product_sentiment_scores = pd.DataFrame({
        'Product': [f'Product {i}' for i in range(1, 21)],
        'Shop Sentiment Score': np.random.rand(20) * 2 - 1,  # Random scores between -1 and 1
        'All Shops Sentiment Score': np.random.rand(20) * 2 - 1   # Random scores between -1 and 1
    })

    # Calculate the gap in sentiment scores
    product_sentiment_scores['Sentiment Gap'] = abs(product_sentiment_scores['Shop Sentiment Score'] - product_sentiment_scores['All Shops Sentiment Score'])

    # Get the top 10 products with the largest gap in sentiment score
    top_products_gap = product_sentiment_scores.nlargest(10, 'Sentiment Gap')

    # Display the table
    st.subheader("Top 10 Products with the Largest Gap in Sentiment Score")
    st.dataframe(top_products_gap[['Product', 'Shop Sentiment Score', 'All Shops Sentiment Score', 'Sentiment Gap']])
