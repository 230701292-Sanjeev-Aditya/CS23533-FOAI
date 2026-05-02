import streamlit as st
import requests
import os
import time
import json

# --- Configuration ---
# 1. FIXED: Use a stable model name to avoid 404 errors.
MODEL_NAME = "gemini-1.5-flash" 
# 2. FIXED: API endpoint for the chosen model.
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

# Fetch API Key from environment
API_KEY = os.environ.get("GEMINI_API_KEY", "")

def safe_api_call(payload):
    """
    Helper to call the API safely without leaking the API Key in logs or UI.
    """
    try:
        response = requests.post(
            API_URL,
            headers={'Content-Type': 'application/json'},
            params={'key': API_KEY},
            json=payload,
            timeout=60
        )
        if response.status_code == 429:
            st.error("⚠️ Rate limit reached. The AI is busy. Please wait 60 seconds and try again.")
            return None
        response.raise_for_status()
        return response.json()
    except Exception as e:
        # 3. FIXED: Catch errors without displaying the URL (leaking the key)
        st.error("An error occurred while connecting to the AI service. Please verify your API Key and connection.")
        return None

@st.cache_data(ttl=3600)
def get_sidebar_data(api_key_check):
    """
    4. FIXED: Combined news and amendments into ONE call to save quota.
    """
    if not api_key_check:
        return "API Key Missing", "API Key Missing"
    
    combined_query = "1. List the latest 3 significant Indian Constitutional Amendments. 2. List the top 2 legal news headlines in India today."
    
    payload = {
        "contents": [{"parts": [{"text": combined_query}]}],
        "tools": [{"google_search": {}}],
        "systemInstruction": {"parts": [{"text": "Summarize briefly. Format clearly."}]}
    }
    
    result = safe_api_call(payload)
    if result:
        text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
        return text
    return "Data temporarily unavailable."

def call_gemini_api(topic, article, query):
    if not API_KEY:
        st.error("Gemini API Key is missing. Please set the GEMINI_API_KEY environment variable.")
        return None, []

    system_prompt = (
        "You are LawXplorer, a specialized AI assistant for the Indian Constitution. "
        "Structure your response into: 1. ANALYSIS and 2. DEFENSE POINTS."
    )

    context = f"Topic: {topic}. Article/Section: {article if article else 'General'}"
    payload = {
        "contents": [{"parts": [{"text": f"{context}\n\nQuery: {query}"}]}],
        "tools": [{"google_search": {}}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }

    result = safe_api_call(payload)
    if result:
        candidate = result.get('candidates', [{}])[0]
        generated_text = candidate.get('content', {}).get('parts', [{}])[0].get('text', '')
        
        sources = []
        grounding_metadata = candidate.get('groundingMetadata', {})
        if grounding_metadata and grounding_metadata.get('groundingAttributions'):
            sources = [
                {'uri': attr['web']['uri'], 'title': attr['web']['title']}
                for attr in grounding_metadata['groundingAttributions'] if attr.get('web')
            ]
        return generated_text, sources
    
    return None, []

# --- UI Logic ---
def main():
    st.set_page_config(page_title="LawXplorer", layout="wide")
    st.title("⚖️ LawXplorer: Law and Compliance Assistant")

    with st.sidebar:
        st.header("Updates")
        # Only attempt to load if key exists to prevent 429 spam on empty keys
        if API_KEY:
            sidebar_content = get_sidebar_data(API_KEY)
            st.markdown(sidebar_content)
        else:
            st.warning("Set GEMINI_API_KEY to see updates.")
        
        st.markdown("---")
        st.caption("DISCLAIMER: Not a substitute for legal advice.")

    # Input Section
    col1, col2 = st.columns(2)
    with col1:
        topic = st.selectbox("Area of Law", ["Fundamental Rights", "Judiciary", "Amendments", "Other"])
    with col2:
        article = st.text_input("Article (Optional)")

    query = st.text_area("Your Question", height=150)

    if st.button("Analyze & Get Guidance", type="primary"):
        if not query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Analyzing legal sources..."):
                response_text, sources = call_gemini_api(topic, article, query)

            if response_text:
                st.subheader("Analysis")
                st.markdown(response_text)
                
                if sources:
                    st.markdown("---")
                    st.subheader("Sources")
                    for source in sources:
                        st.markdown(f"- [{source['title']}]({source['uri']})")
                st.balloons()

if __name__ == "__main__":
    main()
