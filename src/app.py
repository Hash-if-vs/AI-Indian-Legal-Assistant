import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from streamlit_option_menu import option_menu
from initializer import sumllm, qa, pipeline
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from PyPDF2 import PdfReader
import PyPDF2.errors
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
import pdfplumber
from typing import Callable, List, Tuple, Dict
from langchain.docstore.document import Document
import PyPDF2
import re


def conversational_chat(chain, query):
    if "petition" in query:
        prompt = f"""<INST>#Instruction: You are a legal advisor, give services to the clients like drafting petitions, clearing doubts and providing legal assistance according to their queries
                #client:{query}
                #Answer: </INST>"""
        res = pipeline(prompt)
        trim = res[0]["generated_text"].split("</INST>")
        output = trim[1]
        st.session_state["history"].append((query, output))
    else:
        prompt = f"""#Instruction: You are a legal advisor, give services to the clients like drafting petitions, clearing doubts and providing legal assistance according to their queries
                        #client:{query}
                        #Answer: """

        result = chain(
            {"question": prompt, "chat_history": st.session_state["history"]}
        )
        st.session_state["history"].append((query, result["answer"]))
        output = result["answer"]
    return output


def conversational_chat2(chain, query):
    result = chain({"question": query, "chat_history": st.session_state["history2"]})
    st.session_state["history2"].append((query, result["answer"]))
    output = result["answer"]
    return output


def chat2(chain):
    with st.empty().container():

        if "history2" not in st.session_state:
            st.session_state["history2"] = []
        if "generated2" not in st.session_state:
            st.session_state["generated2"] = ["Hello how can I assist you?"]
        if "past2" not in st.session_state:
            st.session_state["past2"] = ["Hey!"]

        response_container = st.container()
        container = st.container()
        with container:
            with st.form(key="my_form", clear_on_submit=True):
                user_input = st.text_input(
                    "query:", placeholder="Type in here...", key="input"
                )
                submit_button = st.form_submit_button(label="chat")
            if submit_button and user_input:
                output = conversational_chat2(chain, user_input)

                st.session_state["past2"].append(user_input)
                st.session_state["generated2"].append(output)
        if st.session_state["generated2"]:
            with response_container:
                for i in range(len(st.session_state["generated2"])):
                    message(
                        st.session_state["past2"][i],
                        is_user=True,
                        key=str(i) + "_user",
                        avatar_style="big-smile",
                    )
                    message(
                        st.session_state["generated2"][i],
                        key=str(i),
                        avatar_style="thumbs",
                    )


def chat(chain):
    with st.empty().container():

        if "history" not in st.session_state:
            st.session_state["history"] = []
        if "generated" not in st.session_state:
            st.session_state["generated"] = ["Hello how can I assist you?"]
        if "past" not in st.session_state:
            st.session_state["past"] = ["Hey!"]
        st.title("Legal assistant")
        response_container = st.container()
        container = st.container()
        with container:
            with st.form(key="my_form", clear_on_submit=True):
                user_input = st.text_input(
                    "query:", placeholder="Type in here...", key="input"
                )
                submit_button = st.form_submit_button(label="chat")
            if submit_button and user_input:
                output = conversational_chat(chain, user_input)

                st.session_state["past"].append(user_input)
                st.session_state["generated"].append(output)
        if st.session_state["generated"]:
            with response_container:
                for i in range(len(st.session_state["generated"])):
                    message(
                        st.session_state["past"][i],
                        is_user=True,
                        key=str(i) + "_user",
                        avatar_style="big-smile",
                    )
                    message(
                        st.session_state["generated"][i],
                        key=str(i),
                        avatar_style="thumbs",
                    )


def home():
    background_image = """
    <style>
    [data-testid="stAppViewContainer"] > .main {
        background-image: url("https://lh3.googleusercontent.com/pw/AP1GczNg67lujMgcHYeTVH3oUhDvFJ9s-kl7RYhxQSIHfbAsQmkh8hPaUCTx3-5O7IVZjy1RICX1NyH1qzg5q4sHxf_1fhtkvG0dAHTb0RKBrxfXfrmVfrnnpdBjVJ-skB4QBW1oCEV4THn0BDhbMzbxrdvIrg5oin8LQ-8JUDKX-2oowUxfzSZnYrGTeXF8EBNQ66k1yuRE36_rScGZrBli_wfkg2Pg0jO0VxTwk4PkmJSAKRpRGTEVXHECgGdLDaHA6JKb34mROZuvrDUZ3hZ-c87P1-_6LJo2bedbb6xhPthUAapdVHcdWv9zk96mgOaBpaXRNLNYxnJV0OqMaVKtmIctQu_Iyze_XDyjmzFGlMk-iIPkhcjQn9MKBarGxeACYIZP6tySSytJuKP5AXF10ZNs99nx97RDPn8ieaK3GiycD8IAPLD-n8SucqI7Gp1xAW6PI4u2ANugpyJcgSsOFGbxgNfwToYyon4JMk2uPNfE_xfZryon13aCaA2FVXUxwKk7cjvb8ggmaYleR2LC0ga4GsKSYE8J6dFtuY-ukAQlSlj7vkDhty3y2uqQK6kcbNl3hISrkcS_ejk2uLH5j-ft2XH6qiFbfwtKmVCuiqwxKoZ1-bOetf0IDVCPJPLrafDn4Y2j9p5CM3vpjqjRyYiXN3zjt4gU9fCy9Goet3dJHxe-Lxgj0OoXpQ_YBSSswMd2NXFvobMgeM2JohMx12-_FSMD00KViXjOF6GuQguZLGXh8oaqw2PgXaq59or_q144bLpjkts5oKgJTIY9KaC15S4lQ6phtM9t_V2v6SdO-R-f0cn99Mhdbokr21p4s5cGvsms07e7FiEYlQ3fTXG_jZIRVU7jsrZSwhF7eZN--Ae2rrjMUSfE38oqiBJKp1mXsZFkNMQ-xZNGvvNfBGU=w1000-h625-s-no-gm?authuser=0");
        background-size: 90vw 100vh;  # This sets the size to cover 100% of the viewport width and height
        background-position: center;  
        background-repeat: no-repeat;
    }
    </style>
    """

    st.markdown(background_image, unsafe_allow_html=True)

    # Your app content goes here
    st.title("Welcome to Legal Assistant")
    st.write("Your virtual legal companion for quick assistance.")
    st.header("Why Legal Knowledge is Necessary:")
    st.write(
        "Legal knowledge is essential for individuals and businesses alike to navigate the complexities of the legal system. Here are a few reasons why it's crucial:"
    )

    st.markdown(
        "- **Compliance:** Understanding legal requirements helps ensure compliance with laws and regulations, avoiding penalties and legal liabilities."
    )
    st.markdown(
        "- **Risk Mitigation:** Knowledge of legal principles enables individuals and businesses to identify and mitigate legal risks, protecting their interests."
    )
    st.markdown(
        "- **Protection of Rights:** Legal knowledge empowers individuals to assert their rights and interests effectively, whether in contracts, disputes, or other legal matters."
    )
    st.markdown(
        "- **Business Operations:** For entrepreneurs and businesses, legal knowledge is vital for structuring operations, managing contracts, and safeguarding intellectual property."
    )

    # How Legal Knowledge is Empowering
    st.header("How Legal Knowledge is Empowering:")
    st.write(
        "Beyond mere compliance, legal knowledge empowers individuals and businesses in various ways:"
    )

    st.markdown(
        "- **Confidence:** With a solid understanding of the law, individuals and businesses can approach legal matters with confidence, knowing their rights and obligations."
    )
    st.markdown(
        "- **Decision-Making:** Legal knowledge enables informed decision-making, guiding choices in business strategies, transactions, and risk management."
    )
    st.markdown(
        "- **Advocacy:** Armed with legal knowledge, individuals can advocate for themselves effectively in legal proceedings, negotiations, and interactions with authorities."
    )
    st.markdown(
        "- **Innovation:** Understanding legal frameworks fosters innovation by providing clarity on intellectual property rights, licensing, and regulatory landscapes."
    )

    # Features section
    st.header("What we Offer")
    st.markdown(
        "- **Legal Assistance** Get answers to your legal quries. Get legal guidance from the legal assistant chatbot. Get empowered by legal Knowledge."
    )
    st.markdown(
        "- **Judgement Summarisation:** Get crisp summaries for court judgements."
    )
    st.markdown("- **Legal Document Drafting:** Generate legal documents with ease.")
    st.markdown(
        "- **Legal Research:** Upload your legal documents and chat with it, speed up your legal research."
    )
    st.markdown("---")


def read_credentials(filename):
    with open(filename, "r") as file:
        credentials = json.load(file)
    return credentials


def write_credentials(filename, new_username, new_password):
    credentials = read_credentials(filename)
    credentials["users"].append({"username": new_username, "password": new_password})
    with open(filename, "w") as file:
        json.dump(credentials, file, indent=4)


def authenticate(username, password):
    # Check if the username exists and if the password matches
    user_credentials = read_credentials("/kaggle/working/project/credentials.json")
    for i in range(len(user_credentials["users"])):
        if username == user_credentials["users"][i]["username"] and user_credentials[
            "users"
        ][i]["password"] == str(password):
            return True
    return False


def is_logged_in():
    global status  # Use global status variable
    return status


def login():
    background_image = """
    <style>
    [data-testid="stAppViewContainer"] > .main {
        background-image: url("https://lh3.googleusercontent.com/pw/AP1GczNg67lujMgcHYeTVH3oUhDvFJ9s-kl7RYhxQSIHfbAsQmkh8hPaUCTx3-5O7IVZjy1RICX1NyH1qzg5q4sHxf_1fhtkvG0dAHTb0RKBrxfXfrmVfrnnpdBjVJ-skB4QBW1oCEV4THn0BDhbMzbxrdvIrg5oin8LQ-8JUDKX-2oowUxfzSZnYrGTeXF8EBNQ66k1yuRE36_rScGZrBli_wfkg2Pg0jO0VxTwk4PkmJSAKRpRGTEVXHECgGdLDaHA6JKb34mROZuvrDUZ3hZ-c87P1-_6LJo2bedbb6xhPthUAapdVHcdWv9zk96mgOaBpaXRNLNYxnJV0OqMaVKtmIctQu_Iyze_XDyjmzFGlMk-iIPkhcjQn9MKBarGxeACYIZP6tySSytJuKP5AXF10ZNs99nx97RDPn8ieaK3GiycD8IAPLD-n8SucqI7Gp1xAW6PI4u2ANugpyJcgSsOFGbxgNfwToYyon4JMk2uPNfE_xfZryon13aCaA2FVXUxwKk7cjvb8ggmaYleR2LC0ga4GsKSYE8J6dFtuY-ukAQlSlj7vkDhty3y2uqQK6kcbNl3hISrkcS_ejk2uLH5j-ft2XH6qiFbfwtKmVCuiqwxKoZ1-bOetf0IDVCPJPLrafDn4Y2j9p5CM3vpjqjRyYiXN3zjt4gU9fCy9Goet3dJHxe-Lxgj0OoXpQ_YBSSswMd2NXFvobMgeM2JohMx12-_FSMD00KViXjOF6GuQguZLGXh8oaqw2PgXaq59or_q144bLpjkts5oKgJTIY9KaC15S4lQ6phtM9t_V2v6SdO-R-f0cn99Mhdbokr21p4s5cGvsms07e7FiEYlQ3fTXG_jZIRVU7jsrZSwhF7eZN--Ae2rrjMUSfE38oqiBJKp1mXsZFkNMQ-xZNGvvNfBGU=w1000-h625-s-no-gm?authuser=0");
        background-size: 90vw 100vh;  # This sets the size to cover 100% of the viewport width and height
        background-position: center;  
        background-repeat: no-repeat;
    }
    </style>
    """

    st.markdown(background_image, unsafe_allow_html=True)

    st.header("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate(username, password):
            st.success("Login successful!")
            st.session_state["status"].append("logged in")

            # Redirect to the chatbot page
        else:
            st.write("Invalid User Name and Password!")
            st.write("Create an account if you are new")


def create_account(full_name, username, password):

    # Save the user's credentials
    # Print the details for now
    print("New Account Created:")
    print("Full Name:", full_name)
    print("Username:", username)
    print("Password:", password)
    write_credentials("/kaggle/working/project/credentials.json", username, password)


def create():
    with st.empty().container():
        background_image = """
    <style>
    [data-testid="stAppViewContainer"] > .main {
        background-image: url("https://lh3.googleusercontent.com/pw/AP1GczNg67lujMgcHYeTVH3oUhDvFJ9s-kl7RYhxQSIHfbAsQmkh8hPaUCTx3-5O7IVZjy1RICX1NyH1qzg5q4sHxf_1fhtkvG0dAHTb0RKBrxfXfrmVfrnnpdBjVJ-skB4QBW1oCEV4THn0BDhbMzbxrdvIrg5oin8LQ-8JUDKX-2oowUxfzSZnYrGTeXF8EBNQ66k1yuRE36_rScGZrBli_wfkg2Pg0jO0VxTwk4PkmJSAKRpRGTEVXHECgGdLDaHA6JKb34mROZuvrDUZ3hZ-c87P1-_6LJo2bedbb6xhPthUAapdVHcdWv9zk96mgOaBpaXRNLNYxnJV0OqMaVKtmIctQu_Iyze_XDyjmzFGlMk-iIPkhcjQn9MKBarGxeACYIZP6tySSytJuKP5AXF10ZNs99nx97RDPn8ieaK3GiycD8IAPLD-n8SucqI7Gp1xAW6PI4u2ANugpyJcgSsOFGbxgNfwToYyon4JMk2uPNfE_xfZryon13aCaA2FVXUxwKk7cjvb8ggmaYleR2LC0ga4GsKSYE8J6dFtuY-ukAQlSlj7vkDhty3y2uqQK6kcbNl3hISrkcS_ejk2uLH5j-ft2XH6qiFbfwtKmVCuiqwxKoZ1-bOetf0IDVCPJPLrafDn4Y2j9p5CM3vpjqjRyYiXN3zjt4gU9fCy9Goet3dJHxe-Lxgj0OoXpQ_YBSSswMd2NXFvobMgeM2JohMx12-_FSMD00KViXjOF6GuQguZLGXh8oaqw2PgXaq59or_q144bLpjkts5oKgJTIY9KaC15S4lQ6phtM9t_V2v6SdO-R-f0cn99Mhdbokr21p4s5cGvsms07e7FiEYlQ3fTXG_jZIRVU7jsrZSwhF7eZN--Ae2rrjMUSfE38oqiBJKp1mXsZFkNMQ-xZNGvvNfBGU=w1000-h625-s-no-gm?authuser=0");
        background-size: 90vw 100vh;  # This sets the size to cover 100% of the viewport width and height
        background-position: center;  
        background-repeat: no-repeat;
    }
    </style>
    """
        st.markdown(background_image, unsafe_allow_html=True)
        st.header("Create Account")
        full_name = st.text_input("Full Name")
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        if st.button("Create Account"):
            create_account(full_name, new_username, new_password)
            st.success("Account created successfully! Please login.")


def info():
    background_image = """
    <style>
    [data-testid="stAppViewContainer"] > .main {
        background-image: url("https://lh3.googleusercontent.com/pw/AP1GczNg67lujMgcHYeTVH3oUhDvFJ9s-kl7RYhxQSIHfbAsQmkh8hPaUCTx3-5O7IVZjy1RICX1NyH1qzg5q4sHxf_1fhtkvG0dAHTb0RKBrxfXfrmVfrnnpdBjVJ-skB4QBW1oCEV4THn0BDhbMzbxrdvIrg5oin8LQ-8JUDKX-2oowUxfzSZnYrGTeXF8EBNQ66k1yuRE36_rScGZrBli_wfkg2Pg0jO0VxTwk4PkmJSAKRpRGTEVXHECgGdLDaHA6JKb34mROZuvrDUZ3hZ-c87P1-_6LJo2bedbb6xhPthUAapdVHcdWv9zk96mgOaBpaXRNLNYxnJV0OqMaVKtmIctQu_Iyze_XDyjmzFGlMk-iIPkhcjQn9MKBarGxeACYIZP6tySSytJuKP5AXF10ZNs99nx97RDPn8ieaK3GiycD8IAPLD-n8SucqI7Gp1xAW6PI4u2ANugpyJcgSsOFGbxgNfwToYyon4JMk2uPNfE_xfZryon13aCaA2FVXUxwKk7cjvb8ggmaYleR2LC0ga4GsKSYE8J6dFtuY-ukAQlSlj7vkDhty3y2uqQK6kcbNl3hISrkcS_ejk2uLH5j-ft2XH6qiFbfwtKmVCuiqwxKoZ1-bOetf0IDVCPJPLrafDn4Y2j9p5CM3vpjqjRyYiXN3zjt4gU9fCy9Goet3dJHxe-Lxgj0OoXpQ_YBSSswMd2NXFvobMgeM2JohMx12-_FSMD00KViXjOF6GuQguZLGXh8oaqw2PgXaq59or_q144bLpjkts5oKgJTIY9KaC15S4lQ6phtM9t_V2v6SdO-R-f0cn99Mhdbokr21p4s5cGvsms07e7FiEYlQ3fTXG_jZIRVU7jsrZSwhF7eZN--Ae2rrjMUSfE38oqiBJKp1mXsZFkNMQ-xZNGvvNfBGU=w1000-h625-s-no-gm?authuser=0");
        background-size: 90vw 100vh;  # This sets the size to cover 100% of the viewport width and height
        background-position: center;  
        background-repeat: no-repeat;
    }
    </style>
    """

    st.markdown(background_image, unsafe_allow_html=True)
    st.header("About")
    st.write(
        "Legal Assistant is designed to streamline legal processes and provide quick access to legal knowledge. \
    Whether you're a lawyer, law student, or just curious about legal matters, \
    our platform aims to simplify legal research and assistance."
    )

    # Contact section
    st.header("Contact Us:")
    st.write(
        "Have questions or suggestions? Feel free to reach out to us at hashifvs0075@gmail.com"
    )
    st.header("Contributors of this project:")
    st.write("Hashif V S, Contact: hashifvs0075@gmail.com")
    st.write("Aleena James, Contact: aleenakjames@gmail.com")
    st.write("Priyan V, Contact: priyanvasantha2@gmail.com")
    st.write("Jomal P Joy, Contact: jomalpjoy@gmail.com")
    st.write("Neena Joseph, Contact: neenajoseph@sjcetpalai.ac.in")
    st.write(
        "Source code is available at https://github.com/Hash-if-vs/Legal-Assistant-Chatbot"
    )

    st.markdown("---")


def summarise():
    background_image = """
    <style>
    [data-testid="stAppViewContainer"] > .main {
        background-image: url("https://lh3.googleusercontent.com/pw/AP1GczNg67lujMgcHYeTVH3oUhDvFJ9s-kl7RYhxQSIHfbAsQmkh8hPaUCTx3-5O7IVZjy1RICX1NyH1qzg5q4sHxf_1fhtkvG0dAHTb0RKBrxfXfrmVfrnnpdBjVJ-skB4QBW1oCEV4THn0BDhbMzbxrdvIrg5oin8LQ-8JUDKX-2oowUxfzSZnYrGTeXF8EBNQ66k1yuRE36_rScGZrBli_wfkg2Pg0jO0VxTwk4PkmJSAKRpRGTEVXHECgGdLDaHA6JKb34mROZuvrDUZ3hZ-c87P1-_6LJo2bedbb6xhPthUAapdVHcdWv9zk96mgOaBpaXRNLNYxnJV0OqMaVKtmIctQu_Iyze_XDyjmzFGlMk-iIPkhcjQn9MKBarGxeACYIZP6tySSytJuKP5AXF10ZNs99nx97RDPn8ieaK3GiycD8IAPLD-n8SucqI7Gp1xAW6PI4u2ANugpyJcgSsOFGbxgNfwToYyon4JMk2uPNfE_xfZryon13aCaA2FVXUxwKk7cjvb8ggmaYleR2LC0ga4GsKSYE8J6dFtuY-ukAQlSlj7vkDhty3y2uqQK6kcbNl3hISrkcS_ejk2uLH5j-ft2XH6qiFbfwtKmVCuiqwxKoZ1-bOetf0IDVCPJPLrafDn4Y2j9p5CM3vpjqjRyYiXN3zjt4gU9fCy9Goet3dJHxe-Lxgj0OoXpQ_YBSSswMd2NXFvobMgeM2JohMx12-_FSMD00KViXjOF6GuQguZLGXh8oaqw2PgXaq59or_q144bLpjkts5oKgJTIY9KaC15S4lQ6phtM9t_V2v6SdO-R-f0cn99Mhdbokr21p4s5cGvsms07e7FiEYlQ3fTXG_jZIRVU7jsrZSwhF7eZN--Ae2rrjMUSfE38oqiBJKp1mXsZFkNMQ-xZNGvvNfBGU=w1000-h625-s-no-gm?authuser=0");
        background-size: 90vw 100vh;  # This sets the size to cover 100% of the viewport width and height
        background-position: center;  
        background-repeat: no-repeat;
    }
    </style>
    """

    st.markdown(background_image, unsafe_allow_html=True)
    st.title("Summarize")
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    if pdf:
        pdf_reader = PdfReader(pdf)
        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text()
        cleanedtext = text.replace("\n", "").replace("\n\n", "")
        st.write(len(cleanedtext.split(" ")))
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        API_URL = (
            "https://api-inference.huggingface.co/models/allenai/led-large-16384-arxiv"
        )
        headers = {"Authorization": "Bearer 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"}

        prompt_template2 = """ Instruction: Create a concise summary of the given document capturing the main points and themes.Please read the provided Original section to understand the context and content.
        Ensure that your final output is thorough, and accurately reflects the document's content and purpose.the content is given below.please generate in less than 150 words
            user: {content}
            Answer:"""
        prompt = PromptTemplate(input_variables=["content"], template=prompt_template2)
        llm_chain = LLMChain(llm=sumllm, prompt=prompt)
        output = llm_chain.run(cleanedtext[:6500])

        st.header("Gnereated summary")
        st.subheader(output)


database = []


def extract_metadata_from_pdf(file_path: str) -> dict:
    try:

        reader = PdfReader(file_path)  # Change this line
        docmetadata = reader.metadata
        return {"title": str(docmetadata.title)}
    except PyPDF2.errors.PdfReadError as e:
        print(f"Error reading PDF file: {e}")


def extract_pages_from_pdf(file_path: str) -> List[Tuple[int, str]]:
    """
    Extracts the text from each page of the PDF.

    :param file_path: The path to the PDF file.
    :return: A list of tuples containing the page number and the extracted text.
    """

    with pdfplumber.open(file_path) as pdf:
        pages = []
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text.strip():  # Check if extracted text is not empty
                pages.append((page_num + 1, text))
    return pages


def parse_pdf(file_path) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
    """
    Extracts the title and text from each page of the PDF.

    :param file_path: The path to the PDF file.
    :return: A tuple containing the title and a list of tuples with page numbers and extracted text.
    """

    metadata = extract_metadata_from_pdf(file_path)
    pages = extract_pages_from_pdf(file_path)

    return pages, metadata


def merge_hyphenated_words(text: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def fix_newlines(text: str) -> str:
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)


def remove_multiple_newlines(text: str) -> str:
    return re.sub(r"\n{2,}", "\n", text)


def clean_text(
    pages: List[Tuple[int, str]], cleaning_functions: List[Callable[[str], str]]
) -> List[Tuple[int, str]]:
    cleaned_pages = []
    for page_num, text in pages:
        for cleaning_function in cleaning_functions:
            text = cleaning_function(text)
        cleaned_pages.append((page_num, text))
    return cleaned_pages


def text_to_docs(text, metadata: Dict[str, str]) -> List[Document]:
    """Converts list of strings to a list of Documents with metadata."""
    doc_chunks = []
    for page_num, page in text:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=200,
        )
        chunks = text_splitter.split_text(page)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "page_number": page_num,
                    "chunk": i,
                    "source": f"p{page_num}-{i}",
                    **metadata,
                },
            )
            doc_chunks.append(doc)

    return doc_chunks


def research():
    st.title("Chat with your documents:")

    pdf = st.file_uploader("Upload your PDF", type="pdf")
    if pdf:
        raw_pages, metadata = parse_pdf(pdf)

        # Step 2: Create text chunks
        cleaning_functions = [
            merge_hyphenated_words,
            fix_newlines,
            remove_multiple_newlines,
        ]
        cleaned_text_pdf = clean_text(raw_pages, cleaning_functions)
        document_chunks = text_to_docs(cleaned_text_pdf, metadata)
        database.extend(document_chunks)

        # Optional: Reduce embedding cost by only using the first 23 pages
        model_name = "BAAI/bge-base-en"
        encode_kwargs = {
            "normalize_embeddings": True
        }  # set True to compute cosine similarity
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name, encode_kwargs=encode_kwargs
        )
        # Step 3 + 4: Generate embeddings and store them in DB
        vector_store = Chroma.from_documents(
            database,
            embeddings,
            persist_directory="db2",
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        sumqa = ConversationalRetrievalChain.from_llm(
            sumllm,
            retriever=retriever
            # memory=memory
        )
        with st.empty().container():
            chat2(sumqa)


def main():

    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=[
                "Home",
                "login",
                "Create Account",
                "Chat",
                "Summarise",
                "Research",
                "About",
            ],
            icons=[
                "house",
                "key",
                "person-add",
                "chat-dots",
                "file-earmark-arrow-up",
                "file",
                "info",
            ],
        )
    if selected == "Home":
        with st.empty().container():
            home()
    if selected == "login":
        with st.empty().container():
            login()
    if selected == "Create Account":
        with st.empty().container():
            create()
    if selected == "Chat":
        with st.empty().container():
            chat(qa)
            pass

    if selected == "Summarise":
        with st.empty().container():
            summarise()

    if selected == "Research":
        with st.empty().container():
            research()
    if selected == "About":
        info()


if __name__ == "__main__":
    main()
