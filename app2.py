import streamlit as st
from streamlit_chat import message
import tempfile
from langchain.llms import HuggingFacePipeline
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from torch import cuda, bfloat16
import transformers
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

model_name = "BAAI/bge-base-en"
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
model_norm = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs
    )
persist_directory = '/content/drive/MyDrive/db'
embedding = model_norm
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})
def load_llm():
    llm=CTransformers(model="TheBloke/Llama-2-7B-Chat-GGUF",
        model_type="llama",
        config={'max_new_tokens': 1024,
                'temperature': 0.1,
                'context_length': 8192,
                })
    return llm
llm=load_llm()
qa = ConversationalRetrievalChain.from_llm(
llm,
retriever=retriever)

def conversational_chat(query):
    result = qa({"question": query,"chat_history":st.session_state['history']})
    st.session_state['history'].append((query,result["answer"]))
    return result['answer']

if 'history' not in st.session_state:
    st.session_state['history']=[]
if 'generated' not in st.session_state:
    st.session_state['generated']=["Hello how can I assist you?"]
if 'past' not in st.session_state:
    st.session_state['past']=["Hey!"]
st.title("Legal assistant")
response_container=st.container()
container=st.container()
with container :
    with st.form(key="my_form",clear_on_submit=True):
        user_input= st.text_input('query:',placeholder="Type in here...",key='input')
        submit_button= st.form_submit_button(label='chat')
    if submit_button and user_input:
        output= conversational_chat(user_input)

        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True,key=str(i)+'_user',avatar_style="big-smile")
            message(st.session_state['generated'][i],key=str(i),
                    avatar_style='thumbs')
            
    

        


  

