import pandas as pd
from sentiment_analysis.gpt_sentiment import load_finbert, gpt_sentiment_score
import os
import json

# Load sample data
df = pd.read_csv("data_ingestion/sample_headlines.csv")

# Load FinBERT model
finbert = load_finbert()

# Your OpenAI key (use env variable or paste directly for now)
OPENAI_KEY = os.getenv("OPENAI_API_KEY") or "your-openai-key-here"

results = []

for idx, row in df.iterrows():
    headline = row['headline']
    print(f"\n🔍 Analyzing: {headline}")

    # FinBERT
    fb_score = finbert(headline)[0]
    print(f"📊 FinBERT: {fb_score['label']} (confidence: {round(fb_score['score']*100, 2)}%)")

    # GPT
    gpt_json = gpt_sentiment_score(headline, OPENAI_KEY)
    try:
        parsed = json.loads(gpt_json)
    except Exception:
        parsed = {"sentiment": "Parsing error", "confidence": 0, "rationale": gpt_json}

    print(f"🤖 GPT: {parsed['sentiment']} | Confidence: {parsed['confidence']}%")
    print(f"🧠 Rationale: {parsed['rationale']}")

    results.append({
        "headline": headline,
        "finbert_sentiment": fb_score['label'],
        "finbert_confidence": round(fb_score['score'] * 100, 2),
        "gpt_sentiment": parsed['sentiment'],
        "gpt_confidence": parsed['confidence'],
        "gpt_rationale": parsed['rationale']
    })

# Save results
output_path = "sentiment_analysis/sentiment_results.csv"
pd.DataFrame(results).to_csv(output_path, index=False)
print(f"\n✅ Sentiment analysis complete. Results saved to {output_path}")
