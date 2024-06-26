import os
import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

# 1. Vectorize the Word Document Data
file_paths = ["Hackathon Spec Cleanup Data.docx", "PowerIndex Data.docx", "Software Options Data.docx", "Spec Performance Data.docx"]

documents = []
for file_path in file_paths:
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        loader = Docx2txtLoader(file_path=file_path)
        documents.extend(loader.load())

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search
def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array

# 3. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0, model="gpt-4o")

template = """
Your name is CPQ BotSensei.
You are a world-class business development representative of our CPQ Application.

I will share a prospect's message with you, and you will provide the most accurate response based on your knowledge of our business specifications.

Below is a message I received from the prospect:
{message}

Here is a list of content realted to this query that we have found in our specifications:
{best_practice}

Please write the most accurate response based on our business specifications:

Please ensure that your response is accurate and aligned with our business specifications..
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

# 4. Retrieval augmented generation
def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response

# 5. Build an app with Streamlit
def main():
    st.set_page_config(page_title="CPQ BotSensei", page_icon=":male-scientist:")

    st.header("CPQ BotSensei :male-scientist:")
    
    # Option to input message or upload a file
    upload_option = st.radio("Choose input method:", ("Text Input", "Upload File"))
    
    if upload_option == "Text Input":
        message = st.text_area("customer message")
    else:
        uploaded_file = st.file_uploader("Choose a file", type=["log"])
        if uploaded_file is not None:
            message = uploaded_file.read().decode("utf-8")
        else:
            message = ""

    if st.button("Send"):
        if message:
            st.write("Generating best practice message...")
            result = generate_response(message)
            st.info(result)
        else:
            st.warning("Please enter a message or upload a file before sending.")

if __name__ == '__main__':
    main()
