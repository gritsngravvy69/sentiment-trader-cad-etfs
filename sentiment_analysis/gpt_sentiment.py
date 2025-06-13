import openai
import os

def gpt_sentiment_score(text, api_key):
    openai.api_key = api_key

    prompt = f"""
    Analyze the sentiment of the following financial news headline. Return your answer in this JSON format:
    {{
        "sentiment": "Positive" | "Negative" | "Neutral",
        "confidence": 0-100,
        "rationale": "Brief explanation of why you scored it this way"
    }}

    Headline: "{text}"
    """

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a financial market sentiment expert trained in analyzing news for ETF trading."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content
