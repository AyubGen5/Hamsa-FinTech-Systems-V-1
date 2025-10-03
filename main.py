import os
import pandas as pd
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables (for API keys etc.)
load_dotenv()

# Load LLM pipeline
generator = pipeline('text-generation', model='gpt2')  # Replace 'gpt2' with your preferred model

# 1. Financial Q&A
def financial_qa(question):
    response = generator(question, max_length=50, num_return_sequences=1)
    return response[0]['generated_text']

# 2. Report Generation
def generate_report(data):
    prompt = f"Generate a financial summary report for the following data: {data}"
    response = generator(prompt, max_length=100, num_return_sequences=1)
    return response[0]['generated_text']

# 3. Risk Analysis
def risk_analysis(data):
    prompt = f"Analyze the financial risk based on this data: {data}"
    response = generator(prompt, max_length=80, num_return_sequences=1)
    return response[0]['generated_text']

# 4. Transaction Categorization
def categorize_transaction(transaction):
    prompt = f"Categorize this transaction: {transaction}"
    response = generator(prompt, max_length=30, num_return_sequences=1)
    return response[0]['generated_text']

# 5. Chatbot for Customer Service
def customer_chatbot(user_query):
    prompt = f"You are a helpful FinTech assistant. {user_query}"
    response = generator(prompt, max_length=50, num_return_sequences=1)
    return response[0]['generated_text']

# 6. Portfolio Recommendation
def portfolio_recommendation(profile):
    prompt = f"Recommend an investment portfolio for: {profile}"
    response = generator(prompt, max_length=70, num_return_sequences=1)
    return response[0]['generated_text']

if __name__ == "__main__":
    # Demo calls
    print("Financial Q&A:", financial_qa("What is compound interest?"))
    print("Report Generation:", generate_report("2023 revenues: $1M, expenses: $600K"))
    print("Risk Analysis:", risk_analysis("Portfolio: Tech stocks, Bonds"))
    print("Transaction Categorization:", categorize_transaction("Payment to Starbucks $4.50"))
    print("Customer Chatbot:", customer_chatbot("How do I reset my password?"))
    print("Portfolio Recommendation:", portfolio_recommendation("Age: 35, Risk: Medium, Goals: Retirement"))