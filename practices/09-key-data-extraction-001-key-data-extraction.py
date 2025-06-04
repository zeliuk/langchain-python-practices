import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1-nano")



from typing import List, Optional
from pydantic import BaseModel, Field

class Person(BaseModel):
    """Information about a person."""

    # ^ Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Person,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    name: Optional[str] = Field(default=None, description="Nombre de la persona")
    lastname: Optional[str] = Field(
        default=None, description="Apellido de la persona si se conoce. Normalmente va seguido del nombre. Por ejemplo Gabriela 'Vargas', Pedro 'García'"
    )
    country: Optional[str] = Field(
        default=None, description="País de la persona si se conoce. Puede ser que nombre alguna región, pero deberías dar el país al cual hace referencia."
    )
    


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        ("human", "{text}"),
    ]
)

'''
chain = prompt | llm.with_structured_output(schema=Person)

comment = "¡Me encanta este producto! Ha sido una revolución en mi rutina diaria. La calidad es excelente y el servicio al cliente es excepcional. Se lo he recomendado a todos mis amigos y familiares. - Gabriela Vargas, Porto Alegre (Río Grande do Sul)."

response = chain.invoke({"text": comment})

print("\n----------\n")

print("Extracción de datos:")

print("\n----------\n")
print(response)

print("\n----------\n")
'''


class Gente(BaseModel):
    """Extracted data about people."""

    # Creates a model so that we can extract multiple entities.
    people: List[Person]
    
chain = prompt | llm.with_structured_output(schema=Gente)

comment = "¡Me encanta este producto! Ha sido una revolución en mi rutina diaria. La calidad es excelente y el servicio al cliente es excepcional. Se lo he recomendado a todos mis amigos y familiares. - Gabriela Vargas, Porto Alegre (Río Grande do Sul)."

response = chain.invoke({"text": comment})

print("\n----------\n")

print("Extracción de datos:")

print("\n----------\n")
print(response)

print("\n----------\n")


# Example input text that mentions multiple people
text_input = """
Alice Johnson, de Canadá, reseñó recientemente un libro que le encantó. Mientras tanto, Bob Smith, de EE. UU., compartió sus opiniones sobre el mismo libro en otra reseña. Ambas fueron muy reveladoras.
"""

# Invoke the processing chain on the text
response = chain.invoke({"text": text_input})

# Output the extracted data
print("\n----------\n")

print("Extracción de datos de una review con varios usuarios:")

print("\n----------\n")
print(response)

print("\n----------\n")

