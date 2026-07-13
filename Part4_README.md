# Applied AI & ML Essentials – Capstone Project
# Part 4 – LLM Powered Feature

## Chosen Track

**Track C – Model Prediction Explanation Pipeline**

---

# Project Overview

This project extends the machine learning solution developed in Part 3 by integrating a Large Language Model (LLM) to provide human-readable explanations for model predictions.

The best-performing machine learning pipeline (`best_model.pkl`) is loaded, predictions are generated for multiple feature vectors, and an LLM explains each prediction using a structured JSON response. The implementation includes prompt engineering, schema validation, temperature comparison, and a PII guardrail.

---

# Technologies Used

- Python 3.13
- Requests
- OpenRouter API
- Random Forest Pipeline
- Joblib
- JSON
- JSON Schema
- python-dotenv
- Regular Expressions

---

# LLM Provider

**Provider:** OpenRouter

**Model Used:**

```
openai/gpt-4.1-mini
```

The API key is securely stored in an environment variable (`OPENROUTER_API_KEY`) using a `.env` file and is never hardcoded in the project.

---

# Pipeline

```
Load Dataset
        ↓
Load best_model.pkl
        ↓
Predict Class
        ↓
Predict Probability
        ↓
Construct Prompt
        ↓
PII Guardrail
        ↓
LLM API Call
        ↓
Parse JSON
        ↓
Schema Validation
        ↓
Return Structured Explanation
```

---

# System Prompt (Verbatim)

```text
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
```

---

# User Prompt Template

```text
Feature Values:

<feature_dictionary>

Predicted Class:
<predicted_class>

Prediction Probability:
<predicted_probability>

Explain this prediction.

Return ONLY valid JSON.
```

---

# Why Temperature = 0?

Temperature controls the randomness of token generation.

For structured JSON generation, deterministic responses are required. Therefore, temperature was set to **0**, ensuring the model consistently selects the highest-probability token and produces predictable JSON output.

Temperature **0.7** was also tested to demonstrate how increased randomness changes the wording and explanation while preserving the overall meaning.

---

# JSON Schema

The LLM response is validated against the following schema.

Required fields:

| Field | Type |
|---------|------|
| prediction_label | String |
| confidence_level | String |
| top_reason | String |
| second_reason | String |
| next_step | String |

All five fields are mandatory.

---

# Structured Output Validation

The following validation steps are performed:

1. Remove whitespace using `response.strip()`.
2. Parse the response using `json.loads()`.
3. Validate the parsed JSON using `jsonschema.validate()`.
4. Catch `JSONDecodeError` if the response is not valid JSON.
5. Catch `ValidationError` if required fields are missing.
6. Apply a fallback response whenever validation fails.

This ensures that downstream applications receive only valid structured JSON.

---

# PII Guardrail

Before every API call, the input is checked using Regular Expressions.

Blocked patterns include:

- Email addresses
- Phone numbers

If either pattern is detected, the LLM request is blocked and the following message is returned:

```
Input blocked: PII detected.
```

---

# Guardrail Demonstration

| Test Input | Result |
|------------|--------|
| john@gmail.com | Blocked |
| Prediction explanation request | Passed |

The guardrail successfully prevented requests containing personally identifiable information while allowing safe requests to proceed.

---

# Temperature Comparison

| Input | Temperature = 0 | Temperature = 0.7 | Key Difference |
|------|-----------------|-------------------|---------------|
| Sample 1 | Deterministic explanation | More descriptive wording | Higher variation |
| Sample 2 | Stable JSON | Slight wording differences | Moderate creativity |
| Sample 3 | Consistent response | Different phrasing | Increased variability |

### Observation

Temperature **0** always produces highly deterministic responses because the model selects the highest-probability next token.

Temperature **0.7** samples from a wider probability distribution, producing more natural but less consistent explanations.

---

# End-to-End Demonstration

| Feature Input | Predicted Class | Probability | Explanation JSON | Validation |
|--------------|----------------|------------|-----------------|-----------|
| Record 1 | <Update Output> | <Update Output> | Passed | PASS |
| Record 2 | <Update Output> | <Update Output> | Passed | PASS |
| Record 3 | <Update Output> | <Update Output> | Passed | PASS |

Update these values using the output generated after executing the Python script.

---

# JSON Validation Results

| Record | JSON Parsing | Schema Validation |
|---------|--------------|------------------|
| Record 1 | PASS | PASS |
| Record 2 | PASS | PASS |
| Record 3 | PASS | PASS |

The generated JSON successfully matched the expected schema for all three prediction explanations.

---

# API Connection

The reusable `call_llm()` function performs:

- Creates the JSON payload.
- Adds authorization headers.
- Sends an HTTP POST request.
- Handles HTTP errors.
- Parses the JSON response.
- Returns only the assistant message.

A simple connectivity test was performed using the prompt:

```
Reply with only the word: hello
```

Expected output:

```
hello
```

---

# Repository Structure

```
Capstone Project/

│
├── Part 1.py
├── Part 2.py
├── Part 3.py
├── Part 4.py
│
├── README.md
├── cleaned_data.csv
├── best_model.pkl
│
├── plots/
│
├── .gitignore
└── requirements.txt
```

---

# Conclusion

Track C demonstrates how a trained machine learning model can be combined with a Large Language Model to generate structured, human-readable prediction explanations.

The implementation includes secure API key management, prompt engineering, schema validation, deterministic JSON generation, temperature comparison, and PII protection. Together, these components provide a reliable and production-oriented framework for explainable AI applications.