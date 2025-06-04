import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

#llm = ChatOpenAI(model="gpt-4.1-nano")

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class Classification(BaseModel):
    sentiment: str = Field(..., enum=["happy", "neutral", "sad"])
    political_tendency: str = Field(
        ...,
        description="The political tendency of the user",
        enum=["conservative", "liberal", "independent"],
    )
    language: str = Field(
        ..., enum=["spanish", "english"]
    )
    
tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)

llm = ChatOpenAI(temperature=0, model="gpt-4.1-nano").with_structured_output(
    Classification
)

tagging_chain = tagging_prompt | llm

feijoo_follower = "Estoy convencido de que el liderazgo sereno y firme de Alberto Núñez Feijóo es justo lo que necesita España. Su apuesta por la estabilidad económica, la unidad territorial y la moderación política puede devolvernos el rumbo y recuperar la confianza de los ciudadanos. Es momento de restaurar el sentido común en la Moncloa."
sanchez_follower = "Creo que el enfoque progresista y valiente de Pedro Sánchez está marcando la diferencia en este país. Su compromiso con los derechos sociales, la igualdad y la sostenibilidad nos está llevando hacia una España más justa e inclusiva. Debemos seguir avanzando con políticas que pongan a las personas en el centro."

response = tagging_chain.invoke({"input": feijoo_follower})

print("\n----------\n")

print("Seguidor de Feijoo:")

print("\n----------\n")
print(response)

print("\n----------\n")

response = tagging_chain.invoke({"input": sanchez_follower})

print("\n----------\n")

print("Seguidor de Pedro:")

print("\n----------\n")
print(response)

print("\n----------\n")