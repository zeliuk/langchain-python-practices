import warnings
from langchain._api import LangChainDeprecationWarning
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)

import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

chatbot = ChatOpenAI(model="gpt-4.1-nano")


from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.memory import FileChatMessageHistory

memory = ConversationBufferMemory(
    chat_memory=FileChatMessageHistory("messages.json"),
    memory_key="messages",
    return_messages=True
)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=chatbot,
    prompt=prompt,
    memory=memory
)

response = chain.invoke("¡hola!")

print("\n----------\n")

print("¡hola!")

print("\n----------\n")
print(response)

print("\n----------\n")

response = chain.invoke("Mi color preferido es el Naranja.")

print("\n----------\n")

print("Mi color preferido es el Naranja.")

print("\n----------\n")
print(response)

print("\n----------\n")

response = chain.invoke("¿Cuál es mi color preferido?")

print("\n----------\n")

print("¿Cuál es mi color preferido?")

print("\n----------\n")
print(response)

print("\n----------\n")
