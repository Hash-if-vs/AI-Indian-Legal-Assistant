from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Together
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from transformers import StoppingCriteria, StoppingCriteriaList
import torch
from torch import cuda, bfloat16
import transformers

chatmodel = "Hashif/Indian-legal-Llama-2-7b-v2"
model_name = "BAAI/bge-base-en"
encode_kwargs = {"normalize_embeddings": True}  # set True to compute cosine similarity
model_norm = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs={"device": "cuda"}, encode_kwargs=encode_kwargs
)


def load_pipeline(model_name):
    model_id = model_name
    device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"
    # set quantization configuration to load large model with less GPU memory
    # this requires the `bitsandbytes` library
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16,
    )

    # begin initializing HF items, you need an access token
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        # use_auth_token=hf_auth
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map="auto",
        # use_auth_token=hf_auth
    )
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        # use_auth_token=hf_auth
    )
    stop_list = ["\nHuman:", "\n```\n", "\n\n"]

    stop_token_ids = [tokenizer(x)["input_ids"] for x in stop_list]
    stop_token_ids
    stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]

    class StopOnTokens(StoppingCriteria):
        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
        ) -> bool:
            for stop_ids in stop_token_ids:
                if torch.eq(input_ids[0][-len(stop_ids) :], stop_ids).all():
                    return True
            return False

    stopping_criteria = StoppingCriteriaList([StopOnTokens()])
    generate_text = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,  # langchain expects the full text
        task="text-generation",
        # we pass model parameters here too
        stopping_criteria=stopping_criteria,  # without this model rambles during chat
        temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=2048,
        repetition_penalty=1.1,  # without this output begins repeating
    )
    return generate_text


def make_chain(llm):

    persist_directory = "/kaggle/working/vectordb2"
    vectordbs = Chroma(
        persist_directory=persist_directory, embedding_function=model_norm
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vectordbs.as_retriever(search_kwargs={"k": 3})
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever
        # memory=memory
    )
    return qa


def make_sumllm():
    llm = Together(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        temperature=0.7,
        max_tokens=1500,
        top_k=1,
        together_api_key="insert your together api key here",
    )
    return llm


sumllm = make_sumllm()
pipeline = load_pipeline(chatmodel)
llm = HuggingFacePipeline(pipeline=pipeline)
qa = make_chain(llm)
