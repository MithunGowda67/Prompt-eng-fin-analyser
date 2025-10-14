import streamlit as st
import json
import re
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from io import BytesIO
import time 

# =================================================================
# 1. STREAMLIT CONFIGURATION (MUST BE FIRST)
# =================================================================

st.set_page_config(
    page_title="Gemini Financial Strategy Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize the Gemini client using the secure st.secrets ---
@st.cache_resource
def get_gemini_client():
    """Initializes and returns the Gemini client using st.secrets."""
    try:
        # st.secrets['gemini']['api_key'] reads the key from .streamlit/secrets.toml
        # This will run immediately after page_config
        return genai.Client(api_key=st.secrets["gemini"]["api_key"])
    except KeyError:
        # We use st.error and st.stop here to handle the critical missing key early
        st.error("Gemini API key not found in `.streamlit/secrets.toml`. Please configure your secrets.")
        st.stop()
    except Exception as e:
        st.error(f"Error initializing Gemini client: {e}")
        st.stop()

# Initialize the client immediately after page_config
client = get_gemini_client()
MODEL_NAME = "gemini-2.5-pro" 


# =================================================================
# 2. PROMPT TEMPLATES AND SCHEMA (Moved to follow configuration)
# =================================================================

# --- Pydantic Schema for Stage 1: Data Extraction ---
class FinancialMetrics(BaseModel):
    """Schema for essential quantitative and qualitative data extracted from a financial report."""
    revenue_current_period: float = Field(description="Total Revenue for the current quarter/year (in millions/billions).")
    revenue_previous_period: float = Field(description="Total Revenue for the previous quarter/year for comparison.")
    net_income_current_period: float = Field(description="Net Income for the current period.")
    net_income_previous_period: float = Field(description="Net Income for the previous period.")
    gross_margin_percentage: float = Field(description="The Gross Margin percentage (e.g., 0.40 for 40%).")
    operating_expense_total: float = Field(description="Total Operating Expenses.")
    cash_from_operations: float = Field(description="Cash Flow from Operating Activities.")
    total_debt: float = Field(description="The company's Total Debt or Liabilities.")
    cash_and_equivalents: float = Field(description="Total Cash and Cash Equivalents on the Balance Sheet.")
    
    management_summary: str = Field(description="The 3-sentence summary of the company's performance from the MD&A section.")
    risk_factors: list[str] = Field(description="List of all major risk factors mentioned in the report.")
    future_outlook_statement: str = Field(description="The most definitive sentence regarding the company's outlook for the next period.")

STAGE_1_SCHEMA = FinancialMetrics.model_json_schema()

# --- Prompt Templates (Moved from prompts.js) ---

STAGE_1_PROMPT = """
You are a highly meticulous Financial Data Analyst. Your sole function is to analyze the provided financial report (PDF, TXT, or MD) and extract the exact values for the specified metrics.

CRITICAL INSTRUCTION: Analyze the entire document, including tables, figures, and narrative text, to locate the requested data points. Provide the values as pure numbers (e.g., 550.5 for $550.5 Million, 0.40 for 40%).

Your output MUST be a JSON object that strictly adheres to the provided schema. DO NOT add any introductory or explanatory text. If a metric is not found, use 0 for numbers or an empty string for text.
"""

STAGE_2_PROMPT_TEMPLATE = """
<System_Prompt>
You are a Senior Financial Strategist. Your task is to perform a detailed financial analysis based solely on the extracted structured data provided in the <Extracted_Data_JSON> tag. Your analysis must follow a structured, step-by-step reasoning process (Chain-of-Thought) to ensure numerical accuracy before drawing conclusions.
</System_Prompt>

<Extracted_Data_JSON>
{extracted_data_placeholder}
</Extracted_Data_JSON>

<Instructions>
First, complete the required reasoning steps in the <Chain_of_Thought> section. Then, use the output of that reasoning to fill the <Intermediate_Analysis> section.

<Chain_of_Thought>
1. **Growth Calculation:** Calculate the Quarter-over-Quarter Revenue Growth Rate (Formula: (Current Revenue - Previous Revenue) / Previous Revenue * 100%). Use the numbers from the JSON.
2. **Profitability Check:** Calculate the Operating Margin (Net Income / Revenue).
3. **Liquidity Assessment:** Comment on the immediate liquidity trend by comparing Cash vs. Total Debt.
4. **Synthesize Work:** Cross-reference the calculated financial trends (growth/margin) with the Management Summary and Risk Factors. What single, major theme connects the financials to the management commentary?
</Chain_of_Thought>

<Intermediate_Analysis>
1. **Revenue & Growth:** [Summary of growth calculation and trend.]
2. **Profitability:** [Summary of operating margin and what Net Income trend reveals.]
3. **Risk Synthesis:** [Detailed note on how the calculated financial health is impacted by the identified risk factors.]
4. **Work Done Assessment (The "Why"):** [Based on the MD&A summary and financial trends, provide a single paragraph assessing the effectiveness of the *work done* by the company during the period.]
</Intermediate_Analysis>
"""

STAGE_3_PROMPT_TEMPLATE = """
<System_Prompt>
You are the CEO's Chief of Staff. Your final task is to condense the entire analysis (provided in the <Full_Analysis_Data> tag) into a three-part, executive-ready final report.
Tone: Professional, direct, and forward-looking.
Output: The final response MUST use Markdown headings (#) and bullet points. DO NOT use any XML tags in your final output.
</System_Prompt>

<Full_Analysis_Data>
{full_analysis_data_placeholder}
</Full_Analysis_Data>

**FINAL REPORT SECTIONS:**

# Executive Summary
**Max 3 Sentences.** Summarize the period's overall performance, highlighting the main success and the primary challenge/risk.

# Key Insights and Work Assessment
Use bullet points to present three distinct insights. One must specifically assess the effectiveness of the *work done* by the company during the period.

# Strategic Suggestions for Next Period
Use bullet points to provide three distinct, actionable, and measurable suggestions for the management team, logically derived from the insights.
"""


# =================================================================
# 3. CORE API FUNCTION (FIX APPLIED)
# =================================================================

def run_gemini_stage(prompt_template, contents, config=None):
    """Sends a request to the Gemini API and returns the text response."""
    
    parts = []
    if isinstance(contents, list) and contents:
        parts.extend(contents)
    elif contents:
         parts.append(contents)
    
    # FIX: Replaced the incorrect Part.from_text() call with the standard dictionary structure for text content.
    # This avoids the TypeError and ensures the prompt_template is correctly included.
    parts.append({"text": prompt_template})

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=parts,
            config=config or types.GenerateContentConfig(max_output_tokens=4096)
        )
        return response.text
    except Exception as e:
        st.exception(f"API Error during stage execution: {e}")
        return None

# =================================================================
# 4. THE MAIN ANALYSIS CHAIN FUNCTION
# =================================================================

def run_financial_analysis_chain(uploaded_file, file_mime_type):
    """Executes the complete multi-stage prompt chain."""
    
    file_bytes = uploaded_file.read()

    file_part = types.Part.from_bytes(
        data=file_bytes,
        mime_type=file_mime_type
    )

    # --- STAGE 1: Data Extraction (Structured Output) ---
    with st.spinner("Stage 1: Analyzing document and extracting structured data..."):
        time.sleep(0.5) 
        stage1_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=STAGE_1_SCHEMA
        )
        json_output = run_gemini_stage(STAGE_1_PROMPT, file_part, stage1_config)

    if not json_output: return None
    
    try:
        extracted_data_json = json.dumps(json.loads(json_output), indent=2)
    except json.JSONDecodeError:
        st.error("Stage 1 failed: Gemini did not return valid JSON. Check document structure.")
        st.code(json_output, language='json')
        return None
    
    st.success("Stage 1 Complete: Structured Data Extracted.")


    # --- STAGE 2: Analytical Reasoning (CoT) ---
    with st.spinner("Stage 2: Performing Chain-of-Thought calculations and analysis..."):
        time.sleep(0.5) 
        stage2_full_prompt = STAGE_2_PROMPT_TEMPLATE.format(extracted_data_placeholder=extracted_data_json)
        
        # NOTE: Stage 2 prompt is passed as text, contents=[] is an empty list as no file is used here.
        stage2_output = run_gemini_stage(stage2_full_prompt, [])

    if not stage2_output: return None
    st.success("Stage 2 Complete: Analysis Ready.")

    match = re.search(r'(<Chain_of_Thought>[\s\S]*<\/Intermediate_Analysis>)', stage2_output, re.DOTALL)
    intermediate_analysis = match.group(1) if match else stage2_output


    # --- STAGE 3: Synthesis and Actionable Suggestions ---
    with st.spinner("Stage 3: Generating executive report..."):
        time.sleep(0.5) 
        full_analysis_data = f"STAGE 1 DATA:\n{extracted_data_json}\n\nSTAGE 2 ANALYSIS:\n{intermediate_analysis}"
        stage3_full_prompt = STAGE_3_PROMPT_TEMPLATE.format(full_analysis_data_placeholder=full_analysis_data)

        # NOTE: Stage 3 prompt is passed as text, contents=[] is an empty list as no file is used here.
        final_report_markdown = run_gemini_stage(stage3_full_prompt, [])

    if not final_report_markdown: return None
    st.success("✅ Analysis Complete! Report Generated.")

    return final_report_markdown, extracted_data_json, stage2_output


# =================================================================
# 5. STREAMLIT UI LAYOUT (No change needed)
# =================================================================

st.title("Gemini Financial Strategy Dashboard")
st.markdown("---")
st.markdown("""
A **3-Stage Prompt Chaining** application using **Gemini 2.5 Pro** to perform secure, structured analysis of financial reports.
""")

# File Uploader
uploaded_file = st.file_uploader(
    "1. Upload Quarterly/Annual Report (PDF or TXT/MD)",
    type=['pdf', 'txt', 'md', 'text/markdown'], 
    help="Upload a document. For PDFs, Gemini uses multimodal reasoning to analyze charts and text."
)

if uploaded_file:
    # Get the file extension and map to MIME type
    if uploaded_file.type == 'application/pdf':
        mime_type = 'application/pdf'
    else:
        mime_type = 'text/plain' 

    if st.button("▶️ Execute 3-Stage Analysis Chain", use_container_width=True, type="primary"):
        # Run the analysis and get results
        report_data = run_financial_analysis_chain(uploaded_file, mime_type)
        
        if report_data:
            final_report_markdown, extracted_json, stage2_output = report_data
            
            # --- Results Display ---
            st.markdown("---")
            st.header("Generated Deliverable")

            # Final Report Column
            st.subheader("Final Executive Report")
            
            st.markdown(final_report_markdown)

            # Download Button
            st.download_button(
                label="⬇️ Download Full Report (.txt)",
                data=final_report_markdown,
                file_name=f"Executive_Analysis_{uploaded_file.name.split('.')[0]}.txt",
                mime="text/plain"
            )

            # --- Debug/Intermediate Output ---
            with st.expander("Show Intermediate Analysis (Debug)", expanded=False):
                st.subheader("Stage 1: Extracted Metrics (Structured JSON)")
                st.code(extracted_json, language='json')
                
                st.subheader("Stage 2: Chain-of-Thought Reasoning (CoT)")
                st.code(stage2_output, language='markdown')
