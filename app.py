import json
import subprocess
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

def download_10k_filings(ticker, start_year):
    """
    Downloads and extracts 10-K filings for the specified ticker and starting year until the current year.
    This function will clone the EDGAR crawler repository, download the requirements.txt if it's not present,
    set up configurations, and run the download and extract scripts.

    Args:
        ticker (str): The stock ticker symbol for the company (e.g., "AAPL").
        start_year (int): The year from which to start downloading filings.
    """
    from datetime import datetime
    current_year = datetime.now().year
    repo_url = "https://github.com/nlpaueb/edgar-crawler.git"
    repo_dir = "edgar-crawler"
    if not os.path.exists(repo_dir):
        print(f"Cloning the repository from {repo_url}...")
        subprocess.run(["git", "clone", repo_url], check=True)
    os.chdir(repo_dir)
    requirements_file = "requirements.txt"
    if not os.path.exists(requirements_file):
        print(f"Downloading {requirements_file}...")
        subprocess.run(["curl", "-O", "https://raw.githubusercontent.com/nlpaueb/edgar-crawler/master/requirements.txt"], check=True)
    print("Installing required dependencies...")
    config = {
        "download_filings": {
            "start_year": start_year,
            "end_year": current_year,
            "quarters": [1, 2, 3, 4],
            "filing_types": ["10-K"],
            "cik_tickers": [ticker],
            "user_agent": "Your Name (your-email@example.com)",
            "raw_filings_folder": "RAW_FILINGS",
            "indices_folder": "INDICES",
            "filings_metadata_file": "FILINGS_METADATA.csv",
            "skip_present_indices": True
        },
        "extract_items": {
            "raw_filings_folder": "RAW_FILINGS",
            "extracted_filings_folder": "EXTRACTED_FILINGS",
            "filings_metadata_file": "FILINGS_METADATA.csv",
            "filing_types": ["10-K"],
            "include_signature": False,
            "items_to_extract": [],
            "remove_tables": True,
            "skip_extracted_filings": True
        }
    }
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)
    try:
        print(f"Downloading filings for {ticker} from {start_year} to {current_year}...")
        subprocess.run(["python", "download_filings.py"], check=True)
        print(f"Extracting items from filings for {ticker}...")
        subprocess.run(["python", "extract_items.py"], check=True)
        print("Process completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
    finally:
        os.chdir('..')

def extract_all_json_content(folder_path):
    """
    Extracts all content from JSON files in the specified folder, using only the first three parts of the filename.

    Args:
        folder_path (str): Path to the folder containing the JSON files.

    Returns:
        list: A list of dictionaries containing the content of each JSON file.
    """
    extracted_content = []
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return extracted_content
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            try:
                parts = file_name.replace(".json", "").split("_")[:3]
                if len(parts) < 3:
                    print(f"Skipping invalid filename: {file_name}")
                    continue
                cik, filing_type, year = parts
                with open(file_path, 'r', encoding="utf-8", errors="ignore") as f:
                    content = json.load(f)
                content["cik"] = cik
                content["filing_type"] = filing_type
                content["year"] = year
                extracted_content.append(content)
                print(f"Successfully extracted data from {file_name}")
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
    return extracted_content

def analyze_sentiment_with_rating(text):
    model_name = "yiyanghkust/finbert-tone"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.detach().numpy()[0]
    exp_logits = np.exp(logits)
    probabilities = exp_logits / np.sum(exp_logits)
    rating_weights = [3, 5, 1]
    rating = sum(prob * weight for prob, weight in zip(probabilities, rating_weights))
    return rating, probabilities

# Main script
if __name__ == "__main__":
    ticker = input("Enter the stock ticker (e.g., 'GOOG'): ").strip()
    start_year = int(input("Enter the start year (e.g., 2019): ").strip())
    download_10k_filings(ticker, start_year)
    data_folder = r"C:\\Users\\varun\\Documents\\SIGMA\\NLP Financial Analysis\\finstreamlit\\edgar-crawler\\datasets\\EXTRACTED_FILINGS\\10-K"
    data = extract_all_json_content(data_folder)
    print(f"\nExtracted data from {len(data)} files.")
    for i, record in enumerate(data[:3]):
        print(f"\nRecord {i + 1}:")
        print(json.dumps(record, indent=4))
