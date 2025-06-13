import openai
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import torch
import os

# Load FinBERT model
def load_finbert():
    tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Use GPT to interpret sentiment in human-like context
def gpt_sentiment_score(text, openai_key):
    openai.api_key = openai_key

    prompt = f"""
    Analyze the sentiment of the following financial news headline. Return your answer in this JSON format:
    {{
        "sentiment": "Positive" | "Negative" | "Neutral",
        "confidence": 0-100,
        "rationale": "Brief explanation of why you scored it this way"
    }}

    Headline: "{text}"
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a financial market sentiment expert trained in analyzing news for ETF trading."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    content = response['choices'][0]['message']['content']
    return content
