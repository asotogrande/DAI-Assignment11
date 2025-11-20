import os
import json
import random
from pathlib import Path
import streamlit as st
from openai import OpenAI


# -----------------------------
# 1. LOAD Q&A DATA
# -----------------------------
DATA_PATH = Path("Q&A_db_practice.json")

@st.cache_data
def load_qa(path=DATA_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

qa_db = load_qa()


# -----------------------------
# 2. SET UP OPENAI CLIENT
# -----------------------------
# IMPORTANT:
# You must set your API key before running:
#   export OPENAI_API_KEY="sk-xxxx"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if client.api_key is None:
    st.error("❌ OPENAI_API_KEY environment variable is not set.")
    st.stop()


# -----------------------------
# 3. LLM EVALUATION FUNCTION
# -----------------------------
EVAL_INSTRUCTIONS = """
You are grading a student's answer to a machine learning theory question.

You will be given:
- The question
- A reference answer written by the instructor
- The student's answer

Tasks:
1. Compare the student's answer with the reference answer.
2. Identify correct points and missing points.
3. Comment on correctness, completeness, clarity, and misconceptions.
4. Give an integer score from 0 to 100.

Scoring rubric:
- 0–39: mostly incorrect or irrelevant
- 40–69: partially correct with major gaps
- 70–89: mostly correct with minor omissions
- 90–100: complete and accurate

Return ONLY valid JSON:
{
  "score": <int>,
  "feedback": "<short explanation>"
}
"""


def call_llm(prompt: str) -> str:
    """Wrapper to call OpenAI Chat API."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",   # fast, cheap, reliable
        messages=[
            {"role": "system", "content": "You are a fair and rigorous ML exam grader."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def evaluate_answer(question: str, reference: str, student: str) -> dict:
    """Send evaluation request to the LLM and parse JSON output."""

    prompt = f"""{EVAL_INSTRUCTIONS}

Question:
{question}

Reference answer:
{reference}

Student answer:
{student}

Now output the JSON:
"""

    raw = call_llm(prompt).strip()

    # Handle accidental code block formatting
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except Exception:
        return {
            "score": 0,
            "feedback": f"⚠️ Could not parse JSON from model.\nRaw output:\n{raw}"
        }


# -----------------------------
# 4. STREAMLIT APP UI
# -----------------------------
st.set_page_config(page_title="ML Q&A Evaluator", page_icon="🤖", layout="centered")

st.title("🤖 Machine Learning Q&A Evaluator")
st.write("Answer the question below and the model will evaluate your response.")


# Select random question
if "index" not in st.session_state:
    st.session_state.index = random.randrange(len(qa_db))

def new_question():
    st.session_state.index = random.randrange(len(qa_db))


qa = qa_db[st.session_state.index]

st.subheader("📘 Question")
st.write(qa["question"])

student_answer = st.text_area("✏️ Your Answer:", height=180)


if st.button("Evaluate My Answer"):
    if not student_answer.strip():
        st.warning("Please write an answer before submitting.")
    else:
        with st.spinner("Evaluating your answer..."):
            result = evaluate_answer(
                qa["question"],
                qa["answer"],
                student_answer
            )

        st.markdown("---")
        st.subheader("📊 Score")
        st.markdown(f"**{result['score']} / 100**")

        st.subheader("💬 Feedback")
        st.write(result["feedback"])

        # Save to session history
        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            "question": qa["question"],
            "student_answer": student_answer,
            "score": result["score"],
            "feedback": result["feedback"],
        })


if st.button("🔀 New Question"):
    new_question()
    st.rerun()


# -----------------------------
# 5. Display evaluation history
# -----------------------------
if "history" in st.session_state and st.session_state.history:
    st.markdown("---")
    st.subheader("📚 Previous Attempts")

    for i, entry in enumerate(reversed(st.session_state.history)):
        with st.expander(f"Attempt {len(st.session_state.history)-i}: Score {entry['score']}"):
            st.write("**Question:**", entry["question"])
            st.write("**Your Answer:**", entry["student_answer"])
            st.write("**Feedback:**", entry["feedback"])

