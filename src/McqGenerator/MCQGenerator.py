
import os
import json
import pandas as pd
from dotenv import load_dotenv
from src.McqGenerator.Logger import logging
from src.McqGenerator.utils import read_file, get_table_data

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain, LLMChain

# Load environment
load_dotenv()
key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    openai_api_key=key,
    model_name="gpt-3.5-turbo",
    temperature=0.7
)

# ----------------------------- PROMPT 1 -----------------------------
TEMPLATE = """
Text: {text}
You are an expert MCQ maker. Given the above text, your task is to
create a quiz of {number} multiple choice questions for {subject} students in a {tone} tone.
Make sure the questions are not repeated and all questions are derived from the text.
Format your response exactly like the RESPONSE_JSON below. Ensure to make {number} MCQs.

### RESPONSE_JSON
{response_json}
"""

guiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=TEMPLATE
)

quiz_chain = LLMChain(
    llm=llm,
    prompt=guiz_generation_prompt,
    output_key="quiz",
    verbose=True
)

# ----------------------------- PROMPT 2 -----------------------------
TEMPLATE2 = """
You are an expert English grammarian and evaluator. Given a Multiple Choice Quiz for {subject} students,
analyze the complexity of the questions and give a concise evaluation (max 50 words).
If the quiz is not appropriate for the students' analytical ability, suggest updated versions
with suitable tone and clarity.

Quiz_MCQs:
{quiz}

Expert evaluation:
"""

quiz_evaluation_prompt = PromptTemplate(
    input_variables=["subject", "quiz"],
    template=TEMPLATE2
)

review_chain = LLMChain(
    llm=llm,
    prompt=quiz_evaluation_prompt,
    output_key="review",
    verbose=True
)

# ----------------------------- CHAIN TOGETHER -----------------------------
generate_evaluate_chain = SequentialChain(
    chains=[quiz_chain, review_chain],
    input_variables=["text", "number", "subject", "tone", "response_json"],
    output_variables=["quiz", "review"],
    verbose=True,
)

