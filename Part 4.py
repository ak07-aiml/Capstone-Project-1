# ==========================================================
# Applied AI & ML Essentials – Capstone Project
# Part 4 – LLM Powered Feature (Track C)
# ==========================================================

import os
import re
import json
import joblib
import warnings
import requests
import pandas as pd

from dotenv import load_dotenv
from jsonschema import validate, ValidationError

warnings.filterwarnings("ignore")

# ==========================================================
# LOAD DATASET
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(BASE_DIR, "cleaned_data.csv")

df = pd.read_csv(csv_path)

print("Dataset Loaded Successfully")

y_reg = df["Value"]

y_clf = (y_reg > y_reg.median()).astype(int)

X = df.drop(
    columns=["Value", "Value_Log"]
)

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)

from sklearn.model_selection import train_test_split

X_train, X_test, _, _ = train_test_split(
    X,
    y_clf,
    test_size=0.20,
    random_state=42
)

print("=" * 80)
print("PART 4 - LLM POWERED FEATURE")
print("=" * 80)

# ==========================================================
# LOAD API KEY
# ==========================================================

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if api_key is None:
    raise ValueError(
        "OpenRouter API Key not found."
    )

print("✓ API Key Loaded")

# ==========================================================
# OPENROUTER SETTINGS
# ==========================================================

url = "https://openrouter.ai/api/v1/chat/completions"

MODEL = "openai/gpt-4.1-mini"

# ==========================================================
# LLM CALL FUNCTION
# ==========================================================

def call_llm(
    system_prompt,
    user_prompt,
    temperature=0.0,
    max_tokens=512
):

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload
    )

    if response.status_code != 200:

        print("Error:", response.status_code)
        print(response.text)

        return None

    return response.json()["choices"][0]["message"]["content"]

print("\nTesting OpenRouter Connection...\n")

response = call_llm(

    system_prompt="You are a helpful assistant.",

    user_prompt="Reply with only the word: hello"

)

print(response)

# ==========================================================
# PII DETECTION
# ==========================================================

def has_pii(text):

    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"

    phone_pattern = (
        r"\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"
    )

    return bool(
        re.search(email_pattern, text)
        or
        re.search(phone_pattern, text)
    )

print("\nTesting PII Guardrail")

blocked = "John email is john@gmail.com"

clean = "Prediction explanation request"

print(has_pii(blocked))

print(has_pii(clean))

# ==========================================================
# LOAD BEST MODEL
# ==========================================================

print("\n" + "=" * 80)
print("LOADING BEST MODEL")
print("=" * 80)

model = joblib.load("best_model.pkl")

print("✓ best_model.pkl loaded successfully.")

# ==========================================================
# JSON SCHEMA
# ==========================================================

response_schema = {
    "type": "object",
    "properties": {
        "prediction_label": {"type": "string"},
        "confidence_level": {"type": "string"},
        "top_reason": {"type": "string"},
        "second_reason": {"type": "string"},
        "next_step": {"type": "string"}
    },
    "required": [
        "prediction_label",
        "confidence_level",
        "top_reason",
        "second_reason",
        "next_step"
    ]
}

# ==========================================================
# SYSTEM PROMPT
# ==========================================================

system_prompt = """
You are an AI assistant that explains machine learning predictions.

Return ONLY valid JSON.

Output exactly this schema:

{
    "prediction_label":"",
    "confidence_level":"",
    "top_reason":"",
    "second_reason":"",
    "next_step":""
}

Do not include markdown.
Do not include explanations outside JSON.
"""

# ==========================================================
# SAMPLE INPUTS
# ==========================================================

sample_inputs = [
    X_test.iloc[0].to_dict(),
    X_test.iloc[1].to_dict(),
    X_test.iloc[2].to_dict()
]

# ==========================================================
# TEMPERATURE COMPARISON
# ==========================================================

print("\n" + "=" * 80)
print("TEMPERATURE COMPARISON")
print("=" * 80)

results = []

for index, sample in enumerate(sample_inputs, start=1):

    sample_df = pd.DataFrame([sample])

    prediction = model.predict(sample_df)[0]

    probability = model.predict_proba(sample_df)[0][prediction]

    user_prompt = f"""
Feature Values:

{json.dumps(sample, indent=2)}

Predicted Class: {prediction}

Prediction Probability: {probability:.4f}

Explain this prediction.

Return ONLY valid JSON.
"""

    # Guardrail

    if has_pii(user_prompt):

        print("Input blocked: PII detected.")
        continue

    # Temperature = 0

    output0 = call_llm(
        system_prompt,
        user_prompt,
        temperature=0
    )

    # Temperature = 0.7

    output7 = call_llm(
        system_prompt,
        user_prompt,
        temperature=0.7
    )

    print("\nRecord", index)

    print("\nTemperature = 0")

    print(output0)

    print("\nTemperature = 0.7")

    print(output7)

    results.append([
        index,
        output0,
        output7
    ])

    # ==========================================================
# VALIDATE OUTPUT
# ==========================================================

print("\n" + "=" * 80)
print("JSON VALIDATION")
print("=" * 80)

for record in results:

    print(f"\nRecord {record[0]}")

    response = record[1]

    try:

        parsed = json.loads(
            response.strip()
        )

        validate(
            parsed,
            response_schema
        )

        print("Validation: PASS")

    except json.JSONDecodeError as e:

        print("Validation: FAIL")

        print(e)

    except ValidationError as e:

        print("Validation: FAIL")

        print(e)

# ==========================================================
# END TO END DEMO
# ==========================================================

print("\n" + "=" * 80)
print("END TO END DEMONSTRATION")
print("=" * 80)

demo = []

for sample in sample_inputs:

    sample_df = pd.DataFrame([sample])

    prediction = model.predict(sample_df)[0]

    probability = model.predict_proba(sample_df)[0][prediction]

    user_prompt = f"""
Feature Values:

{json.dumps(sample)}

Predicted Class:

{prediction}

Probability:

{probability:.3f}

Explain.
"""

    if has_pii(user_prompt):

        status = "Blocked"

        output = None

    else:

        output = call_llm(
            system_prompt,
            user_prompt,
            temperature=0
        )

        status = "Passed"

    print("\nInput")

    print(sample)

    print("\nOutput")

    print(output)

    print("\nStatus")

    print(status)

    demo.append([
        prediction,
        probability,
        status
    ])

print("\n" + "=" * 80)
print("PART 4 COMPLETED SUCCESSFULLY")
print("=" * 80)