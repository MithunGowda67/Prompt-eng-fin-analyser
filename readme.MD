💎 Gemini Financial Strategy Dashboard
This repository contains a full-stack, secure prototype for analyzing lengthy, complex financial reports (PDFs, TXT) and synthesizing them into executive-level, actionable summaries.

The application uses a secure 3-Stage Prompt Chain methodology to ensure high accuracy, structured data extraction, and deep analytical reasoning, transforming raw document data into strategic insights.

✨ Project Highlights & Key Technologies
Secure API Architecture (Streamlit Secrets): The Gemini API Key is never exposed in the client-side code or GitHub. It is securely managed via Streamlit's st.secrets.

3-Stage Prompt Chaining: A robust workflow breaks the complex analysis task into three sequential, specialized steps to minimize hallucination and maximize output quality.

Multimodal Input: Capable of processing structured tables and unstructured narrative text directly from uploaded PDFs and plaintext documents.

Structured Output: Leverages the pydantic library to enforce a strict JSON schema for Stage 1, ensuring extracted financial metrics are machine-readable and reliable.

Chain-of-Thought (CoT) Reasoning: Stage 2 forces the LLM to perform explicit ratio calculations and cross-reference financial figures with management commentary before drawing conclusions.

🛠️ Project Structure and Methodology
The core solution is contained within a single Python file, making deployment fast and easy.

File

Purpose

Notes

streamlit_app.py

Main Application Logic: Handles UI, file upload, and executes the entire 3-stage prompt chain securely.

Includes all UI, processing, and API logic.

.streamlit/secrets.toml

Security Configuration: Securely stores the gemini.api_key environment variable for Streamlit Cloud.

File must NOT be pushed to GitHub.

requirements.txt

Defines necessary Python packages: streamlit, google-genai, pydantic.

Essential for Streamlit Cloud deployment.

The 3-Stage Prompt Chain Methodology
Stage

Role

Output Format

Goal

1. Data Extraction

Financial Data Analyst

Strict JSON Schema

Securely extracts quantitative figures (Revenue, Debt, Margins) and core qualitative statements directly from the uploaded document.

2. Analytical Reasoning

Senior Financial Strategist

CoT (XML Tags)

Calculates critical ratios (e.g., Growth, Operating Margin) and synthesizes the data with risk factors to determine the 'why' behind the financial results.

3. Final Synthesis

CEO's Chief of Staff

Clean Markdown

Converts the deep analysis into a concise Executive Summary, Key Insights, and actionable Strategic Suggestions, ready for senior management review.

🚀 Secure Deployment on Streamlit Cloud
The recommended deployment path uses Streamlit's built-in security features, ensuring your API key is never exposed.

1. Prerequisites
Python 3.9+

Node Package Manager (npm) (for local testing only)

A Gemini API Key (free tier is sufficient for testing)

A GitHub Account

2. Deployment Steps
Local Secret Setup: Create the .streamlit folder and add your secrets.toml file with your key. Crucially, add .streamlit/secrets.toml to your .gitignore file.

Commit and Push: Ensure your streamlit_app.py and requirements.txt are pushed to a public GitHub repository.

Deploy to Cloud: Go to the [Streamlit Community Cloud] website, select "New app," and connect your repository.

Inject Secret: In the deployment configuration's Secrets section, manually paste the contents of your local secrets.toml file. This securely injects the key into the running application's environment.

The app will deploy, and the key will remain hidden from the public.