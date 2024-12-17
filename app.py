import streamlit as st
import pandas as pd
from groq import Groq
from typing import Generator
import concurrent.futures
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ...existing code...

client = Groq(api_key=st.secrets["groq_api_key"])
models = "llama-3.3-70b-versatile"
st.set_page_config(page_icon="💬", layout="wide",
                   page_title="Personal Assistant")

st.subheader("My Personal Assistant", divider="rainbow", anchor=False)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

def fetch_response(query):
    """Fetch response from Groq API for a single query."""
    prompt = """
    You are a Principal Solution Architect with over 30 years of experience in designing, implementing, and overseeing enterprise-grade solutions. Your expertise spans across security, compliance, IT best practices, and scalable, customer-centric technical solutions. You have a deep understanding of industry standards such as ISO 27001, GDPR, SOC 2, and NIST frameworks. Your goal is to provide responses that instill trust and confidence in clients, ensuring they feel assured that your company is a leader in security, compliance, and solution delivery.

    Tone & Style: Your responses must be clear, precise, and easy to understand, avoiding technical jargon unless necessary. Avoid vague statements and provide actionable, concrete assurances. Frame your responses to position your company's solution as the best option, highlighting unique differentiators.

    Objective: Every answer should resolve the customer's question so completely that no follow-up questions are needed. The response should demonstrate mastery in security, compliance, and IT best practices while emphasizing the company's commitment to customer success, risk mitigation, and operational excellence.

    Response Structure:

    Clear Statement of Assurance: Open with a direct, confident statement that addresses the customer's concern.
    Actionable Explanation: Explain how your company achieves this, referencing specific methods, certifications, or tools.
    Key Differentiator: Highlight a unique benefit, service, or approach that sets your company apart.
    Closing Assurance: End with a confident, trust-building statement that eliminates doubt and reaffirms the company's capability.
    Example Question: How does your company ensure data security and compliance with industry standards?

    Example Response (Using the Prompt Above):

    Our company ensures data security and compliance with industry standards by adhering to internationally recognized frameworks such as ISO 27001, SOC 2, and GDPR.
    We maintain a robust Information Security Management System (ISMS) with continuous risk assessments, incident response planning, and proactive threat detection. Our security protocols include multi-factor authentication (MFA), role-based access control (RBAC), data encryption (both in transit and at rest), and regular penetration testing by independent auditors.
    Unlike many providers, we offer 24/7 security monitoring backed by a dedicated compliance team to address emerging threats in real-time. Our ability to rapidly adapt to new regulatory requirements ensures that our customers remain compliant, even as standards evolve.
    With this multi-layered approach, you can be confident that your data is secure, your compliance obligations are met, and your business is always audit-ready."""
    
    """
    retries = 3
    for attempt in range(retries):
        try:
            system_prompt = {
                "role": "system",
                "content": prompt
            }
            chat_completion = client.chat.completions.create(
                model=models,
                messages=[
                    system_prompt,
                    {"role": "user", "content": query}
                ],
                stream=True
            )
            full_response = ''.join(generate_chat_responses(chat_completion))
            return full_response
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error fetching response for query '{query}': {e}")
            if 'Rate limit reached for model' in error_message and attempt < retries - 1:
                logger.info(f"Rate limit reached. Retrying in 10 seconds... (Attempt {attempt + 1}/{retries})")
                time.sleep(10)
            else:
                return f"Error: {e}"

def main():

    query_type = st.radio("Choose input method:", ("Single Question", "Multiple Queries from File"))

    if query_type == "Single Question":
        query = st.text_area("Enter your query:")
        if st.button("Submit"):
            st.session_state.messages.append({"role": "user", "content": query})

            with st.chat_message("user", avatar='👨‍💻'):
                st.markdown(query)

            # Fetch response from Groq API for the query
            response = fetch_response(query)
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(response)
    else:
        uploaded_file = st.file_uploader("Upload a CSV or XLSX file", type=["csv", "xlsx"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.write("Uploaded file:")
            st.write(df)
            if st.button("Submit"):
                queries = df.iloc[:, 0].tolist()  # Assuming the queries are in the first column

                # Process the queries in parallel
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    responses = list(executor.map(fetch_response, queries))

                # Add responses to the dataframe
                df['Response'] = responses
                st.write("Processed file with responses:")
                st.write(df)
                # Optionally, allow the user to download the processed file
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download processed file",
                    data=csv,
                    file_name='processed_queries.csv',
                    mime='text/csv',
                )

if __name__ == "__main__":
    main()
