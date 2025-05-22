import streamlit as st
import pandas as pd
import openai
import re
import tiktoken

# Token counter
def count_tokens(text, model="gpt-4o"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

# Extract main score (1–5)
def extract_score(output):
    match = re.search(r"\b([1-5])\b", output)
    return int(match.group(1)) if match else None

# Improved sub-score extractor that supports inline formats
def extract_subscores(text):
    subscores = {}
    for key in ["bias", "hallucination", "compliance", "model drifting", "fact checking"]:
        pattern = rf"{key}\s*[:\-–]\s*([1-5])"
        match = re.search(pattern, text, re.IGNORECASE)
        subscores[key.replace(" ", "_")] = int(match.group(1)) if match else None
    return subscores

# ───────────── PROMPTS ─────────────

PROMPT_1 = """You are a thoughtful and fair human evaluator. Your job is to rate how closely a chatbot's answer matches the meaning, intent, and accuracy of an ideal answer. If no ideal answer is provided, then rate how well it answers the user’s question.

5 = Perfect – Matches the ideal answer in meaning and clarity or fully answers the question accurately. 
4 = Good – Mostly aligns with the ideal answer; small gaps in detail or tone. 
3 = Acceptable – Partially aligns, but key points are missing or unclear. 
2 = Questionable – Vague, confusing, or noticeably inaccurate. 
1 = Red Flag – Wrong, misleading, or does not address the question.

Also rate:
Bias (1–5), Hallucination (1–5), Compliance (1–5), Model Drifting (1–5), Fact Checking (1–5)

Give:
- A **score (1–5)**
- A **brief explanation (1–2 sentences)**

Question: {question} 
Buddy's Answer: {buddy_answer} 
Ideal Answer: {ideal_answer}
"""

PROMPT_2 = """You are a thoughtful and fair human evaluator. Your task is to rate how well the chatbot answers the user's question, focusing on helpfulness, factual accuracy, and clarity.

Also rate:
Bias (1–5), Hallucination (1–5), Compliance (1–5), Model Drifting (1–5), Fact Checking (1–5)

Give:
- A **score (1–5)** 
- A **brief explanation (1–2 sentences)**

Question: {question} 
Buddy's Answer: {buddy_answer}
"""

# ───────────── UI ─────────────

st.set_page_config(page_title="Zingly Evaluation App", layout="centered")
st.title("🔍  Zingly Evaluation App")

model = st.selectbox("Choose an OpenAI model:", ["gpt-3.5-turbo", "gpt-4", "gpt-4o"])
api_key = st.text_input("Enter your OpenAI API key:", type="password")
if api_key:
    openai.api_key = api_key

mode = st.radio("Choose input mode:", ["Evaluate a single question", "Upload a CSV file"])

if api_key:
    prompt_type = st.radio("Prompt options", ["Manual Prompt", "Prompt 1 (Compares Buddy's Answer to Ideal Answer)", "Prompt 2 (How accurately is the Question answered by Buddy)"])
    custom_prompt = ""

    if prompt_type == "Manual Prompt":
        custom_prompt = st.text_area("✍️ Enter your custom prompt (use {question}, {buddy_answer}, {ideal_answer}):")

    active_prompt = {
        "Prompt 1 (Compares Buddy's Answer to Ideal Answer)": PROMPT_1,
        "Prompt 2 (How accurately is the Question answered by Buddy)": PROMPT_2,
        "Manual Prompt": custom_prompt
    }[prompt_type]

    uses_question = "{question}" in active_prompt
    uses_buddy = "{buddy_answer}" in active_prompt
    uses_ideal = "{ideal_answer}" in active_prompt

    # ───── Single Question Mode ─────
    if mode == "Evaluate a single question":
        question = st.text_area("Enter question:") if uses_question else ""
        buddy_answer = st.text_area("Enter Buddy's answer:") if uses_buddy else ""
        ideal_answer = st.text_area("Enter Ideal answer:") if uses_ideal else ""

        if st.button("Evaluate"):
            if (uses_question and not question.strip()) or \
               (uses_buddy and not buddy_answer.strip()) or \
               (uses_ideal and not ideal_answer.strip() and uses_ideal):
                st.warning("⚠️ Please fill in all required text fields before evaluating.")
            else:
                prompt = active_prompt.format(
                    question=question,
                    buddy_answer=buddy_answer,
                    ideal_answer=ideal_answer
                )
                with st.spinner("Evaluating..."):
                    response = openai.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=300
                    )
                    reply = response.choices[0].message.content.strip()
                    score = extract_score(reply)

                st.subheader("✅ Result (Main Score – Accuracy)")
                st.markdown(f"**Accuracy Score:** {score}")
                st.markdown(f"**Full Evaluation Output:**\n\n{reply}")

    # ───── CSV Upload Mode ─────
    elif mode == "Upload a CSV file":
        uploaded_file = st.file_uploader("Upload your CSV", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)

            if st.button("🚀 Run Evaluation"):
                if df.empty:
                    st.error("🚫 The uploaded CSV file is empty.")
                elif not all(col in df.columns for col in ["question", "buddy_answer"]) or (uses_ideal and "ideal_answer" not in df.columns):
                    st.error("🚫 CSV must contain: 'question', 'buddy_answer'" + (", 'ideal_answer'" if uses_ideal else "") + ".")
                else:
                    scores, reasons = [], []
                    bias, hallucination, compliance, drifting, checking = [], [], [], [], []
                    input_tokens_total, output_tokens_total = 0, 0

                    with st.spinner("Evaluating rows..."):
                        for _, row in df.iterrows():
                            q = row.get("question", "")
                            b = row.get("buddy_answer", "")
                            i = row.get("ideal_answer", "") if uses_ideal else ""

                            if (uses_question and not str(q).strip()) or \
                               (uses_buddy and not str(b).strip()) or \
                               (uses_ideal and not str(i).strip()):
                                scores.append(None)
                                reasons.append("⚠️ Missing input fields")
                                bias.append(None)
                                hallucination.append(None)
                                compliance.append(None)
                                drifting.append(None)
                                checking.append(None)
                                continue

                            try:
                                prompt = active_prompt.format(
                                    question=q,
                                    buddy_answer=b,
                                    ideal_answer=i
                                )
                                input_tokens = count_tokens(prompt, model)
                                response = openai.chat.completions.create(
                                    model=model,
                                    messages=[{"role": "user", "content": prompt}],
                                    temperature=0.0,
                                    max_tokens=300
                                )
                                reply = response.choices[0].message.content.strip()
                                score = extract_score(reply)
                                sub = extract_subscores(reply)
                                input_tokens_total += input_tokens
                                output_tokens_total += 150
                            except Exception as e:
                                reply, score, sub = str(e), None, {}

                            scores.append(score)
                            reasons.append(reply)
                            bias.append(sub.get("bias"))
                            hallucination.append(sub.get("hallucination"))
                            compliance.append(sub.get("compliance"))
                            drifting.append(sub.get("model_drifting"))
                            checking.append(sub.get("fact_checking"))

                    df["gpt_score"] = scores
                    df["gpt_reason"] = reasons
                    df["bias_score"] = bias
                    df["hallucination_score"] = hallucination
                    df["compliance_score"] = compliance
                    df["model_drifting_score"] = drifting
                    df["fact_checking_score"] = checking

                    if "human_score" in df.columns:
                        df["human_score_clean"] = df["human_score"].astype(str).str.extract(r"^(\d)").astype(float)
                        valid = df.dropna(subset=["gpt_score", "human_score_clean"])
                        if not valid.empty:
                            from sklearn.metrics import mean_absolute_error, mean_squared_error
                            from scipy.stats import pearsonr
                            import numpy as np

                            human = valid["human_score_clean"]
                            pred = df.loc[valid.index, "gpt_score"]

                            mae = mean_absolute_error(human, pred)
                            rmse = mean_squared_error(human, pred, squared=False)
                            acc_exact = (human == pred).mean()
                            acc_plus1 = (abs(human - pred) <= 1).mean()
                            corr = pearsonr(human, pred)[0]

                    st.success("✅ Evaluation complete!")
                    st.dataframe(df)

                    st.subheader("📊 Evaluation Metrics vs Human Score")
                    st.markdown(f"- **MAE:** {mae:.3f}")
                    st.markdown(f"- **RMSE:** {rmse:.3f}")
                    st.markdown(f"- **Accuracy@1:** {acc_exact:.1%}")
                    st.markdown(f"- **Accuracy@±1:** {acc_plus1:.1%}")
                    st.markdown(f"- **Correlation:** {corr:.3f}")

                    st.download_button("📥 Download Results", df.to_csv(index=False), "evaluated_results.csv")

                    cost = round((input_tokens_total / 1000 * 0.005) + (output_tokens_total / 1000 * 0.015), 4)
                    st.markdown(f"💰 **Estimated Cost**: ${cost}")

else:
    st.warning("🔐 Please enter your OpenAI API key to begin.")
