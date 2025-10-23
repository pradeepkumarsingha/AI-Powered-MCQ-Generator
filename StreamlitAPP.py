import os
import json
import pandas as pd
from dotenv import load_dotenv
from src.McqGenerator.utils import read_file, get_table_data
import streamlit as st
from langchain_community.callbacks.manager import get_openai_callback
import traceback
from src.McqGenerator.MCQGenerator import generate_evaluate_chain
from src.McqGenerator.Logger import logging
import re

# Load response template
with open(r"D:/GEN AI/MCQ/experiment/response.json", 'r') as file:
    RESPONSE_JSON = json.load(file)

st.title("MCQs Creator Application with Langchain")

def extract_quiz_json(quiz_str):
    """
    Extracts the JSON portion from the AI response.
    Returns a Python dict or None if extraction fails.
    """
    try:
        if not quiz_str:
            return None
        # Use regex to find JSON object
        match = re.search(r'\{.*\}', quiz_str, re.DOTALL)
        if not match:
            return None
        json_str = match.group(0)
        return json.loads(json_str)
    except Exception as e:
        print("Error parsing quiz JSON:", e)
        return None

with st.form("user_inputs"):
    uploaded_file = st.file_uploader("Upload a PDF or TXT file")
    mcq_count = st.number_input("No. of MCQs", min_value=3, max_value=50)
    subject = st.text_input("Insert a Subject", max_chars=20)
    tone = st.text_input("Complexity level of question", max_chars=20, placeholder="Simple")
    button = st.form_submit_button("Create MCQs")

    if button and uploaded_file and mcq_count and subject and tone:
        with st.spinner("Loading...."):
            try:
                # Read file content
                text = read_file(uploaded_file)

                # Generate MCQs with Langchain
                with get_openai_callback() as cb:
                    response = generate_evaluate_chain({
                        "text": text,
                        "number": mcq_count,
                        "subject": subject,
                        "tone": tone,
                        "response_json": json.dumps(RESPONSE_JSON)
                    })

            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error("Error generating MCQs")

            else:
                # Optional: display token usage
                print(f"Total tokens: {cb.total_tokens}")
                print(f"Prompt tokens: {cb.prompt_tokens}")
                print(f"Completion tokens: {cb.completion_tokens}")
                print(f"Total cost: {cb.total_cost}")

                if isinstance(response, dict):
                    quiz_raw = response.get("quiz", None)
                    if quiz_raw:
                        # Show raw AI output for debugging
                        st.text_area("Raw Quiz JSON", quiz_raw, height=200)

                        # Extract JSON safely
                        quiz_dict = extract_quiz_json(quiz_raw)
                        if quiz_dict:
                            table_data = get_table_data(json.dumps(quiz_dict))
                            if table_data:
                                df = pd.DataFrame(table_data)
                                df.index += 1
                                st.table(df)
                                st.text_area(label="Review", value=response.get("review", ""))
                            else:
                                st.warning("No valid MCQ data found!")
                        else:
                            st.error("Could not extract valid JSON from AI response")
                    else:
                        st.error("Quiz data not found in AI response")
                else:
                    st.write(response)
