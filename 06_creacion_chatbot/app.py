"""
author: Mikel Agirrebeitia
"""

import os
from dotenv import load_dotenv

import streamlit as st
from langchain.prompts import PromptTemplate
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from create_db import VectorDB
vectordb = VectorDB()

# Important: hardcoding the API key in Python code is not a best practice. We are using
# this approach for the ease of demo setup. In a production application these variables
# can be stored in an .env or a properties file

# URL of the hosted LLMs is hardcoded because at this time all LLMs share the same endpoint
url = "https://us-south.ml.cloud.ibm.com"

# These global variables will be updated in get_credentials() functions
watsonx_project_id = ""
# Replace with your IBM Cloud key
api_key = ""

def get_credentials():

    load_dotenv('.env')

    # Update the global variables that will be used for authentication in another function
    globals()["api_key"] = os.getenv("api_key", None)
    globals()["watsonx_project_id"] = os.getenv("project_id", None)

    print("*** Got credentials***")

# The get_model function creates an LLM model object with the specified parameters
def get_model(model_type,max_tokens,min_tokens,decoding,stop_sequences):

    generate_params = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.MIN_NEW_TOKENS: min_tokens,
        GenParams.DECODING_METHOD: decoding,
        GenParams.STOP_SEQUENCES:stop_sequences
    }

    memory = ConversationBufferMemory(
    max_messages=5,
    memory_key="history",
    input_key="question"  # Exclude 'context' here
    )
    retriever = vectordb.retriever
    template = """

    - Devuelve todas las respuestas en formato Markdown, utilizando negrita, cursiva, y enlaces cuando sea necesario.
    - Responde a la pregunta basándote en el contexto proporcionado (delimitado por <ctx></ctx>) y el historial de chat (delimitado por <hs></hs>).
    - Si el contexto está vacio, responde que la pregunta está fuera del alcance.
    - Verifica siempre tu información para evitar cualquier inexactitud. Nunca alucines ni fabriques detalles.

    ------
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {history}
    </hs>
    ------
    {question}
    ---
    Answer:

    """
    prompt = PromptTemplate(
    input_variables=["history", "context", "question"],  # Include 'context' here
    template=template,
    )

    model = Model(
        model_id=model_type,
        params=generate_params,
        credentials={
            "apikey": api_key,
            "url": url
        },
        project_id=watsonx_project_id
        )
    # Create the model objects that will be used by LangChain
    current_llm = WatsonxLLM(model=model)
    qa = RetrievalQA.from_chain_type(
        llm=current_llm,
        chain_type='stuff',
        retriever=retriever,
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
            "memory": memory,
        }
    )
    

    return qa


def answer_questions(question):
    # Get the model
    model = st.session_state.model

    # Generate response
    generated_response = model({"query": question})
    print(generated_response)
    model_output = generated_response['result']
    # For debugging
    print("Answer: " + model_output)

    # Display output on the Web page
    with st.chat_message('assistant'):
        st.markdown(model_output, unsafe_allow_html=True)
    st.session_state.chat_messages.append((model_output, 1))

# Invoke the main function
# answer_questions()

if __name__ == '__main__':
    if st.session_state.get('start') is None:
        st.session_state['start'] = True
        st.session_state.chat_messages = []
        get_credentials()
        model_type = ModelTypes.LLAMA_2_70B_CHAT
        max_tokens = 200
        min_tokens = 100
        decoding = DecodingMethods.GREEDY
        stop_sequences = ['.']
        st.session_state.model = get_model( model_type, max_tokens, min_tokens, decoding, stop_sequences)

    st.title('🌠 Create a chatbot with watsonx.ai LLM')
    for message, sender in st.session_state.chat_messages:
        message_sender = ('user', 'assistant')[sender==1]
        with st.chat_message(message_sender):
            st.markdown(message)

    user_message = st.chat_input('Introduce tu pregunta:')
    if user_message is not None:
        st.chat_message('user').markdown(user_message)
        st.session_state.chat_messages.append((user_message, 0))
        with st.spinner('Buscando respuesta...'):
            answer_questions(user_message)


