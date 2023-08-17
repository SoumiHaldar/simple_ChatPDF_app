import streamlit as st
# from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import pickle
import os
from streamlit_extras.colored_header import colored_header
from streamlit_extras.app_logo import add_logo
from dotenv import load_dotenv
load_dotenv()
openai_api=os.getenv('OPEN_API_KEY')


add_logo("chatgpt.jpg",height=100)
# st.markdown('# This is a Chat App')
with st.sidebar:
    st.header("About")
    st.subheader('''
        This app is built using:
       - [Streamlit](https://streamlit.io/)
       - [Langchain](https://python.langchain.com/)
       - [OpenAI](https://platform.openai.com/docs/models)
                ''' )
with open('style.css') as b:
    st.markdown(f'<style>{b.read()}</style>', unsafe_allow_html=True)


    
def new():
    colored_header(
        label="This is a Chat App",
        description='This app is an LLM-powered chatbot',
        color_name='green-70'
    )
    st.header('Chat with PDF')
    # Upload a pdf
    pdf_file=st.file_uploader('Upload your pdf',type='pdf') 
    if pdf_file is not None:
        pdf_reader=PdfReader(pdf_file)
        # st.write(pdf_reader)

        text=''
        for each_page in pdf_reader.pages:
            text=text+ each_page.extract_text()
        
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks=text_splitter.split_text(text=text)
        # embeddings
        
        file_name=pdf_file.name[:-4]
        if os.path.exists(f'{file_name}.pkl'):
            with open(f'{file_name}.pkl','rb') as f:
                vectorstore= pickle.load(f)           
        else:
            embeddings=OpenAIEmbeddings()
            vectorstore=FAISS.from_texts(chunks,embedding=embeddings)
            with open(f'{file_name}.pkl','wb') as f:
                pickle.dump(vectorstore,f)

        # User questions
        query=st.text_input("Ask questions about your uploaded pdf file")
        with open('style1.css') as c:
            st.markdown(f'<style>{c.read()}</style>', unsafe_allow_html=True)
        if query:
            docs=vectorstore.similarity_search(query=query)
            llm=OpenAI(temperature=0)
            chain=load_qa_chain(llm=llm,chain_type='stuff')
            response=chain.run(input_documents=docs,question=query)
            st.write(response)




if __name__=='__main__':
    new()