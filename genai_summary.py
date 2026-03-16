from pathlib import Path
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env from the same folder as this file
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            f"OPENAI_API_KEY not found. Expected it in environment or in: {env_path}"
        )
    return OpenAI(api_key=api_key)

def generate_procurement_summary(insights, top_variance_df, top_fragment_df):
    client = get_openai_client()

    top_var_items = top_variance_df[['short_text', 'variance_pct']].head(5).to_dict(orient='records')
    top_frag_items = top_fragment_df[['short_text', 'unique_suppliers']].head(5).to_dict(orient='records')

    prompt = f"""
You are a senior procurement consultant. Based on the data below, generate a 5-bullet strategic summary:

- Core insights: {insights}
- Top price variance items: {top_var_items}
- Top fragmented items: {top_frag_items}

Use business-friendly language.
Suggest opportunities or risks.
Avoid technical jargon.
"""

    response = client.chat.completions.create(
        model="gpt-5.1-2025-11-13",
        messages=[
            {"role": "system", "content": "You are a helpful procurement assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content