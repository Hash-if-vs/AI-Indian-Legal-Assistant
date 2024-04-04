import streamlit as st
from streamlit_chat import message
import tempfile
from langchain.llms import HuggingFacePipeline
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
    # Load the locally downloaded model here
    model_id = 'meta-llama/Llama-2-7b-chat-hf'

    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
    bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
                           )

# begin initializing HF items, you need an access token
    hf_auth = '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'
    model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
    )

# enable evaluation mode to allow model inference
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
    )
    stop_list = ['\nHuman:', '\n```\n']

    stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
    stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_ids in stop_token_ids:
                if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                    return True
            return False

    stopping_criteria = StoppingCriteriaList([StopOnTokens()])
    generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model rambles during chat
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,# max number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
        )
    llm = HuggingFacePipeline(pipeline=generate_text)

    
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
            
    

        


  

