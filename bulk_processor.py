import sqlite3
import pandas as pd
import os
import argparse
from config import DB_NAME, get_gemini_model, GEMINI_MODELS, DEFAULT_MODEL
from schemas import ExtractedLegalTest
import google.generativeai as genai

def read_prompt(file_path):
    with open(file_path, 'r') as f:
        return f.read()

def run_extraction_bulk(case_text, api_key, model_name):
    model = get_gemini_model(model_name=model_name, api_key=api_key)
    prompt = read_prompt('prompts/extractor_prompt.txt')
    response = model.generate_content(
        [prompt, case_text],
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json",
            response_schema=ExtractedLegalTest
        )
    )
    return ExtractedLegalTest.parse_raw(response.text)

def bulk_process_cases(api_key: str, model_name: str):
    """Processes all cases in the database that have not been processed yet."""
    conn = sqlite3.connect(DB_NAME)
    
    # Get cases that do not have an entry in legal_tests table
    cases_to_process = pd.read_sql("""
        SELECT c.case_id, c.case_name, c.full_text
        FROM cases c
        LEFT JOIN legal_tests lt ON c.case_id = lt.case_id
        WHERE lt.test_id IS NULL
    """, conn)
    
    if cases_to_process.empty:
        print("No cases to process.")
        conn.close()
        return

    print(f"Found {len(cases_to_process)} cases to process.")

    for index, row in cases_to_process.iterrows():
        print(f"Processing case: {row['case_name']} (ID: {row['case_id']})")
        try:
            extracted_test_obj = run_extraction_bulk(row['full_text'], api_key, model_name)
            
            c = conn.cursor()
            c.execute("INSERT INTO legal_tests (case_id, test_novelty, extracted_test_summary, source_paragraphs, source_type, validator_name) VALUES (?, ?, ?, ?, ?, ?)",
                      (row['case_id'], extracted_test_obj.test_novelty, extracted_test_obj.extracted_test_summary, extracted_test_obj.source_paragraphs, 'ai_extracted_bulk', 'bulk_processor'))
            conn.commit()
            print(f"  Successfully extracted and saved test for {row['case_name']}")
        except Exception as e:
            print(f"  Error processing {row['case_name']}: {e}")
            conn.rollback() # Rollback in case of error

    conn.close()
    print("Bulk processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk process SCC cases for legal test extraction.")
    parser.add_argument("--api_key", type=str, help="Your Gemini API Key. Can also be set via GEMINI_API_KEY environment variable.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL, help="The Gemini model name to use for extraction.")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not provided. Please set it as an environment variable or pass it with --api_key.")
    else:
        bulk_process_cases(api_key, args.model_name)