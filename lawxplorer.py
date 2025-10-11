import streamlit as st
import requests
import os
import time
import json

# --- Configuration ---
# Uses the environment variable GEMINI_API_KEY for security
API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL_NAME = "gemini-2.5-flash-preview-05-20"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

# --- Backend Logic (API Call with Grounding) ---

def call_gemini_api(topic, article, query):
    """
    Calls the Gemini API with Google Search Grounding to get cited legal analysis.
    Implements exponential backoff for resilience.
    """
    # This check remains the primary fail-fast point for the main query
    if not API_KEY:
        st.error("Gemini API Key is missing. Please set the GEMINI_API_KEY environment variable.")
        return None, []

    # 1. Build the system prompt to define the AI's persona
    system_prompt = (
        "You are LawXplorer, a specialized AI assistant focused on the Indian Constitution. "
        "Your task is to provide accurate, concise, and professional legal analysis based on your search results. "
        "Strictly cite all sources used. Summarize the answer in clear, accessible language, and focus primarily on the Indian constitutional context."
    )

    # 2. Build the user prompt
    context = f"Topic: {topic}. Article/Section: {article if article else 'General Inquiry'}"
    user_query = f"Analyze the following legal query within the context of the Indian Constitution: '{query}'"

    # 3. Construct the full payload
    payload = {
        "contents": [{"parts": [{"text": f"{context}\n\n{user_query}"}]}],
        "tools": [{"google_search": {}}],  # Enable Google Search Grounding
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }

    # 4. API Request with Exponential Backoff
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                API_URL,
                headers={'Content-Type': 'application/json'},
                params={'key': API_KEY},
                json=payload,
                timeout=60
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            result = response.json()
            candidate = result.get('candidates', [{}])[0]
            
            # Extract text
            generated_text = candidate.get('content', {}).get('parts', [{}])[0].get('text', 'No response generated.')

            # Extract grounding sources
            sources = []
            grounding_metadata = candidate.get('groundingMetadata', {})
            if grounding_metadata and grounding_metadata.get('groundingAttributions'):
                sources = grounding_metadata['groundingAttributions']
                sources = [
                    {'uri': attr['web']['uri'], 'title': attr['web']['title']}
                    for attr in sources if attr.get('web')
                ]
            
            return generated_text, sources

        except requests.exceptions.HTTPError as e:
            # Handle rate limiting (429) or transient server errors (500, 503)
            if response.status_code in [429, 500, 503] and attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                time.sleep(wait_time)
                continue
            st.error(f"HTTP Error: {e}. Status Code: {response.status_code}")
            return None, []
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred during the API request: {e}")
            return None, []
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            return None, []
            
    return None, []

@st.cache_data(ttl=3600) # Cache the result for 1 hour to avoid excessive calls
def get_amendment_info(api_key_for_cache):
    """
    Fetches the current or most recent constitutional amendment information.
    The API_KEY is explicitly passed to the cached function to ensure it's
    available during the initial run.
    """
    if not api_key_for_cache:
        # This branch should now be hit correctly if API_KEY is missing
        return "Amendment info unavailable (API Key missing)."

    amendment_query = "What is the most recent significant amendment to the Indian Constitution? Provide the amendment number and a one-sentence summary of its impact."
    
    payload = {
        "contents": [{"parts": [{"text": amendment_query}]}],
        "tools": [{"google_search": {}}], 
        "systemInstruction": {"parts": [{"text": "You are a concise legal news summarizer. Answer the user's question with only the amendment number and a brief summary. Do not add salutations or extra context."}]},
    }

    try:
        response = requests.post(
            API_URL,
            headers={'Content-Type': 'application/json'},
            params={'key': api_key_for_cache}, # Use the passed key
            json=payload,
            timeout=10 
        )
        response.raise_for_status()
        
        result = response.json()
        text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'Could not retrieve information.')
        return text
    except requests.exceptions.RequestException as e:
        return f"Request error: Check your API Key or connection. {e}"
    except Exception as e:
        return f"Internal error during parsing: {e}"


# --- Frontend/UI Logic ---

def main():
    st.set_page_config(page_title="LawXplorer: Indian Constitution Assistant", layout="wide")

    st.title("⚖️ LawXplorer: Indian Constitution Assistant")
    st.markdown("A specialized AI tool for legal analysis and guidance on the Indian Constitution.")
    st.markdown("---")

    # Sidebar for API Key check and AMENDMENT INFO
    with st.sidebar:
        st.header("Configuration & Disclaimer")
        if not API_KEY:
            st.warning("Please set the GEMINI_API_KEY environment variable to use the assistant.")
        else:
            st.success("API Key Loaded.")
        
        # New Amendment Information Block
        st.markdown("---")
        st.subheader("💡 Latest Constitutional Amendment")
        
        # FIX: Pass the API_KEY explicitly to the cached function.
        if API_KEY:
             with st.spinner("Fetching current legal updates..."):
                 # Pass the API_KEY as an argument
                 amendment_summary = get_amendment_info(API_KEY)
             st.info(amendment_summary)
        else:
            st.info("Amendment info unavailable (API Key missing).")


        st.markdown(
            "**DISCLAIMER:** LawXplorer provides legally relevant information and cites sources using Google Search grounding. "
            "It is for informational purposes only and is **not a substitute for professional legal advice** from a qualified lawyer."
        )


    st.markdown("### Your Legal Query")

    # Input fields
    col1, col2 = st.columns([1, 1])

    with col1:
        topic = st.selectbox(
            "Area of Law/Topic",
            ["Fundamental Rights", "Directive Principles", "Union & State Relations", "Constitutional Amendments", "Judiciary & Courts", "Other"],
            help="Select the broad area of the query."
        )

    with col2:
        article = st.text_input(
            "Relevant Article/Section (Optional)",
            placeholder="e.g., Article 21, Article 19",
            help="Specify a relevant article for focused research."
        )

    query = st.text_area(
        "Your Specific Legal Question or Compliance Scenario",
        placeholder="e.g., What are the current limitations on the Right to Freedom of Speech under Article 19(1)(a)? Provide recent Supreme Court judgments.",
        height=150
    )

    if st.button("Analyze & Get Guidance", type="primary"):
        if not query:
            st.warning("Please enter your legal question before analyzing.")
        else:
            with st.spinner("LawXplorer is consulting the Constitution and legal sources..."):
                response_text, sources = call_gemini_api(topic, article, query)

            if response_text:
                st.markdown("---")
                st.subheader("LawXplorer Analysis & Guidance")
                
                # Display Analysis
                st.markdown(response_text)
                
                # Display Sources (Citations)
                st.markdown("---")
                st.subheader("📚 Cited Sources (Grounding)")
                
                if sources:
                    source_markdown = ""
                    for i, source in enumerate(sources, 1):
                        source_markdown += f"{i}. [{source['title']}]({source['uri']})\n"
                    st.markdown(source_markdown)
                else:
                    st.info("No external sources were cited for this specific response.")
                
                st.balloons()


if __name__ == "__main__":
    main()
