#%%
from typing import Optional

from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_structured_output_chain,
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage

from pydantic import BaseModel, Field


class Person(BaseModel):
    """Identifying information about a person."""

    name: str = Field(..., description="The person's name")
    age: int = Field(..., description="The person's age")
    fav_food: Optional[str] = Field(None, description="The person's favorite food")

#%%
# If we pass in a model explicitly, we need to make sure it supports the OpenAI function-calling API.
llm = ChatOpenAI(model="gpt-4", temperature=0)

prompt_msgs = [
    SystemMessage(
        content="You are a world class algorithm for extracting information in structured formats."
    ),
    HumanMessage(
        content="Use the given format to extract information from the following input:"
    ),
    HumanMessagePromptTemplate.from_template("{input}"),
    HumanMessage(content="Tips: Make sure to answer in the correct format"),
]
prompt = ChatPromptTemplate(messages=prompt_msgs)

chain = create_structured_output_chain(Person, llm, prompt, verbose=True)
chain.run("Sally is 13")
# %%

from typing import Sequence


class People(BaseModel):
    """Identifying information about all people in a text."""

    people: Sequence[Person] = Field(..., description="The people in the text")


chain = create_structured_output_chain(People, llm, prompt, verbose=True)
chain.run(
    "Sally is 13, Joey just turned 12 and loves spinach. Caroline is 10 years older than Sally, so she's 23."
)

# %%
class OptionalFavFood(BaseModel):
    """Either a food or null."""

    food: Optional[str] = Field(
        None,
        description="Either the name of a food or null. Should be null if the food isn't known.",
    )


def record_person(name: str, age: int, fav_food: OptionalFavFood) -> str:
    """Record some basic identifying information about a person.

    Args:
        name: The person's name.
        age: The person's age in years.
        fav_food: An OptionalFavFood object that either contains the person's favorite food or a null value. Food should be null if it's not known.
    """
    return f"Recording person {name} of age {age} with favorite food {fav_food.food}!"


chain = create_openai_fn_chain([record_person], llm, prompt, verbose=True)
res = chain.run(
    "The most important thing to remember about Tommy, my 12 year old, is that he'll do anything for apple pie."
)


# %%


