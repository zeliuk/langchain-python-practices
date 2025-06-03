import warnings

from langchain._api import LangChainDeprecationWarning

warnings.simplefilter("ignore", category=LangChainDeprecationWarning)

import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

chatbot = ChatOpenAI(model="gpt-4.1-nano")



from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage

chatbotMemory = {}

# input: session_id, output: chatbotMemory[session_id]
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chatbotMemory:
        chatbotMemory[session_id] = ChatMessageHistory()
    return chatbotMemory[session_id]


chatbot_with_message_history = RunnableWithMessageHistory(
    chatbot, 
    get_session_history
)

session1 = {"configurable": {"session_id": "001"}}

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="Mi color favorito es el Naranja")],
    config=session1,
)

print("\n----------\n")

print("Mi color favorito es el Naranja")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="¿Cúal es mi colo favorito?")],
    config=session1,
)

print("\n----------\n")

print("¿Cúal es mi colo favorito? (session1)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

session2 = {"configurable": {"session_id": "002"}}

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="¿Cúal es mi colo favorito?")],
    config=session2,
)

print("\n----------\n")

print("¿Cúal es mi colo favorito? (session2)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

session1 = {"configurable": {"session_id": "001"}}

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="¿Cúal es mi colo favorito?")],
    config=session1,
)

print("\n----------\n")

print("¿Cúal es mi colo favorito? (session1 otra vez)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

session2 = {"configurable": {"session_id": "002"}}

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="Mi nombre es Carlos.")],
    config=session2,
)

print("\n----------\n")

print("Mi nombre es Carlos. (session2)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="¿Cúal es mi nombre?")],
    config=session2,
)

print("\n----------\n")

print("¿Cúal es mi nombre? (session2)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="¿Cúal es mi colo favorito?")],
    config=session1,
)

print("\n----------\n")

print("¿Cúal es mi colo favorito? (session1)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough


def limited_memory_of_messages(messages, number_of_messages_to_keep=2):
    return messages[-number_of_messages_to_keep:]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

limitedMemoryChain = (
    RunnablePassthrough.assign(messages=lambda x: limited_memory_of_messages(x["messages"]))
    | prompt 
    | chatbot
)

chatbot_with_limited_message_history = RunnableWithMessageHistory(
    limitedMemoryChain,
    get_session_history,
    input_messages_key="messages",
)

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="Mi moto favorita es KTM.")],
    config=session1,
)

print("\n----------\n")

print("Mi moto favorita es KTM. (session1)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="Mi ciudad favorita es Barcelona.")],
    config=session1,
)

print("\n----------\n")

print("Mi ciudad favorita es Barcelona. (session1)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

responseFromChatbot = chatbot_with_limited_message_history.invoke(
    {
        "messages": [HumanMessage(content="¿Cuál es mi color favorito?")],
    },
    config=session1,
)

print("\n----------\n")

print("¿Cuál es mi color favorito? (chatbot con memoria limitada a los 3 últimos mensajes)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="¿Cuál es mi color favorito?")],
    config=session1,
)

print("\n----------\n")

print("¿Cuál es mi color favorito? (chatbot con memoria ilimitada)")

print("\n----------\n")
print(responseFromChatbot.content)

print("\n----------\n")