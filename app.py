import streamlit as st
import pandas as pd
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq.chat_models import ChatGroq
from langchain_together import ChatTogether
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_mistralai import ChatMistralAI


from dotenv import load_dotenv
from pandasai import SmartDataframe
#from pandasai import SmartDatalake
from pandasai.llm import BambooLLM
from pandasai import Agent
from pandasai.responses.streamlit_response import StreamlitResponse

import matplotlib.pyplot as plt

import os

# Load environment variables
load_dotenv()

#LANGSMITH VAR
langchain_tracing_v2 = os.getenv('LANGCHAIN_TRACING_V2')
langchain_project = os.getenv('LANGCHAIN_PROJECT')
langchain_endpoint = os.getenv('LANGCHAIN_ENDPOINT')
langchain_api_key = os.getenv('LANGCHAIN_API_KEY')

# If you want to set new values for the environment variables
os.environ['LANGCHAIN_TRACING_V2'] = langchain_tracing_v2
os.environ['LANGCHAIN_PROJECT'] = langchain_project
os.environ['LANGCHAIN_ENDPOINT'] = langchain_endpoint
os.environ['LANGCHAIN_API_KEY'] = langchain_api_key


#Dictionary to store the extracted dataframes
data = {}

def main():

    st.set_page_config(page_title = "PandasAI",page_icon = "🐼")
    st.title("Chat with Your Data using PandasAI:🐼")
    #reading the csv file
    
    #Side Menu Bar
    with st.sidebar:
        st.title("Configuration:⚙️")
        #Activating Demo Data
        st.text("Data Setup: 📝")
        file_upload = st.file_uploader("Upload your Data",accept_multiple_files=False,type = ['csv','xls','xlsx'])

        st.markdown(":green[*Please ensure the first row has the column names.*]")

        #selecting LLM to use
        llm_type = st.selectbox(
                            "Please select LLM",
                            ('Groq','Mistral','BambooLLM','Together','Deepseek','OpenAI','Ollama'),index=0)
        
        #Adding users API Key
        user_api_key = st.text_input('Please add your API key',placeholder='Paste your API key here',type = 'password')
        
        #Get Pandas API key here
        #st.markdown("[Get Your PandasAI API key here](https://www.pandabi.ai/auth/sign-up)")

    if file_upload is not None:
        data  = extract_dataframes(file_upload)
        df = st.selectbox("Here's your uploaded data!",
                          tuple(data.keys()),index=0
                          )
        st.dataframe(data[df])
        
        
        llm = get_LLM(llm_type,user_api_key)

        if llm:
            #Instattiating PandasAI agent
            analyst = get_agent(data,llm)

            #starting the chat with the PandasAI agent
            chat_window(analyst)
            
    else:
        st.warning("Please upload your data first! You can upload a CSV or an Excel file.")

#Function to get LLM
def get_LLM(llm_type,user_api_key):
    #Creating LLM object based on the llm type selected:
    try:
        if llm_type == 'BambooLLM':
            if user_api_key:
                os.environ["PANDASAI_API_KEY"] = user_api_key
            
            else:
                # If no API key provided, try to get it from environment variables
                os.environ["PANDASAI_API_KEY"]= os.getenv('PANDASAI_API_KEY')

            llm = BambooLLM()

        elif llm_type =='Groq':
            if user_api_key:
                os.environ["GROQ_API_KEY"] = user_api_key     
            
            else:
                # Configure the API key
                os.environ["GROQ_API_KEY"]= os.getenv('GROQ_API_KEY')

            llm = ChatGroq(model_name="llama-3.1-70b-versatile", temperature=0, seed=26, api_key = os.environ['GROQ_API_KEY'])

        elif llm_type =='Mistral':
            if user_api_key:
                os.environ["MISTRAL_API_KEY"] = user_api_key     
            
            else:
                # Configure the API key
                os.environ["MISTRAL_API_KEY"]= os.getenv('MISTRAL_API_KEY')
            
            llm = ChatMistralAI(model_name="open-mistral-nemo", temperature=0, seed=26, api_key = os.environ['MISTRAL_API_KEY'])

        elif llm_type =='Together':
            if user_api_key:
                os.environ["TOGETHER_API_KEY"] = user_api_key     
            
            else:
                # Configure the API key
                os.environ["TOGETHER_API_KEY"]= os.getenv('TOGETHER_API_KEY')

            llm = ChatTogether(model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", temperature=0, seed=26, together_api_key = os.environ['TOGETHER_API_KEY'])

        elif llm_type =='Deepseek':
            if user_api_key:
                os.environ["DEEPSEEK_API_KEY"] = user_api_key     
            
            else:
                # Configure the API key
                os.environ["DEEPSEEK_API_KEY"]= os.getenv('DEEPSEEK_API_KEY')
            llm = ChatOpenAI(model="deepseek-coder", temperature=0, seed=26, base_url='https://api.deepseek.com', api_key = os.environ['DEEPSEEK_API_KEY'])

        elif llm_type =='OpenAI':
            if user_api_key:
                os.environ["OPENAI_API_KEY"] = user_api_key     
            
            else:
                # Configure the API key
                os.environ["OPENAI_API_KEY"]= os.getenv('OPENAI_API_KEY')
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, seed=26, api_key = os.environ['OPENAI_API_KEY'])


        elif llm_type =='Ollama':
            if user_api_key:
                os.environ["OPENAI_API_KEY"] = user_api_key     
            
            else:
                # Configure the API key
                os.environ["OLLAMA_API_KEY"]="NOKEY" 
            llm = ChatOpenAI(model="llama3.1:8b", temperature=0, seed=26, base_url="http://127.0.0.1:11434/v1/")
        return llm

    except Exception as e:
        #st.error(e)
        st.error("No/Incorrect API key provided! Please Provide/Verify your API key")

   

#Function for chat window
def chat_window(analyst):
    with st.chat_message("assistant"):
        st.text("Explore your data with PandasAI?🧐")

    #Initilizing message history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    #Displaying the message history on re-reun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Printing the questions
            if 'question' in message:
                st.markdown(message["question"])
            # Printing the code generated and the evaluated code
            elif 'response' in message:
                # Getting the response
                st.write(message["response"])
            # Retrieving error messages
            elif 'error' in message:
                st.text(message['error'])
    if "images" in st.session_state:
        for img in st.session_state.images:
            st.image(img)
    #Getting the questions from the users
    user_question = st.chat_input("What are you curious about? ")
   
    if user_question:
        #Displaying the user question in the chat message
        with st.chat_message("user"):
            st.markdown(user_question)
        #Adding user question to chat history
        st.session_state.messages.append({"role":"user","question":user_question})
       
        try:
            with st.spinner("Analyzing..."):
                response = analyst.chat(user_question)
                explanation = analyst.explain()
                if os.path.exists("exports/charts/temp_chart.png"):
                    im = plt.imread("exports/charts/temp_chart.png")
                    if "images" not in st.session_state:
                        st.session_state.images = []
                    st.session_state.images.append(im)
                    with st.chat_message("assistant"):
                        st.image(im)
                        st.write(response)
                    st.session_state.messages.append({"role":"assistant","response":response})
                    os.remove("exports/charts/temp_chart.png")
                else:
                    st.write(response)
                    st.write(explanation)
                    st.session_state.messages.append({"role":"assistant","response":response})
       
        except Exception as e:
            st.write(e)
            error_message = "⚠️Sorry, Couldn't generate the answer! Please try rephrasing your question!"

    #Function to clear history
    def clear_chat_history():
        st.session_state.messages = []
    #Button for clearing history
    st.sidebar.text("Click to Clear Chat history")
    st.sidebar.button("CLEAR 🗑️",on_click=clear_chat_history)

        
def get_agent(data,llm):
    """
    The function creates an agent on the dataframes exctracted from the uploaded files
    Args: 
        data: A Dictionary with the dataframes extracted from the uploaded data
        llm:  llm object based on the ll type selected
    Output: PandasAI Agent
    """
    agent = Agent(list(data.values()),config = {"llm":llm,"verbose": True, "response_parser": StreamlitResponse})

    return agent

def extract_dataframes(raw_file):
    """
    This function extracts dataframes from the uploaded file/files
    Args: 
        raw_file: Upload_File object
    Processing: Based on the type of file read_csv or read_excel to extract the dataframes
    Output: 
        dfs:  a dictionary with the dataframes
    
    """
    dfs = {}
    if raw_file.name.split('.')[1] == 'csv':
        csv_name = raw_file.name.split('.')[0]
        df = pd.read_csv(raw_file)
        dfs[csv_name] = df

    elif (raw_file.name.split('.')[1] == 'xlsx') or (raw_file.name.split('.')[1] == 'xls') :
        # Read the Excel file
        xls = pd.ExcelFile(raw_file)

        # Iterate through each sheet in the Excel file and store them into dataframes
        dfs = {}
        for sheet_name in xls.sheet_names:
            dfs[sheet_name] = pd.read_excel(raw_file, sheet_name=sheet_name)

    #return the dataframes
    return dfs

if __name__ == "__main__":
    main()
