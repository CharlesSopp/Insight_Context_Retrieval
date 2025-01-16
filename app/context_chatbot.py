from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
import os

load_dotenv("../.env")
AZURE_API_KEY = os.getenv('AZURE_API_KEY')
AZURE_ENDPOINT = os.getenv('AZURE_ENDPOINT')

llm = AzureChatOpenAI(
    azure_deployment='gpt-4o',
    api_version="2024-08-01-preview",
    temperature=0.7,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
)

SYS_PROMPT = """You are a helpful assistant tasked with answering questions about network incidents close notes. These close noters have been summarised for you, and you will answer user queries based on this summary."""

summarise_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT),
        ("human", """Summary of the network incidents' close notes:
                     {summary_text}
                     User query:
                     {user_query}
                  """),
    ]
)

close_notes_chatbot = (summarise_prompt
                            |
                            llm
                            |
                            StrOutputParser())