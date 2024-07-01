#%%
from typing import Iterator, List

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "Write a comma-separated list of 5 animals similar to: {animal}"
)
model = ChatOpenAI(temperature=0.0)

str_chain = prompt | model | StrOutputParser()

# %%
for chunk in str_chain.stream({"animal": "bear"}):
    print(chunk, end="", flush=True)
# %%
def split_into_list(input: Iterator[str]) -> Iterator[List[str]]:
    # hold partial input until we get a comma
    buffer = ""
    for chunk in input:
        # add current chunk to buffer
        buffer += chunk
        # while there are commas in the buffer
        while "," in buffer:
            # split buffer on comma
            comma_index = buffer.index(",")
            # yield everything before the comma
            yield [buffer[:comma_index].strip()]
            # save the rest for the next iteration
            buffer = buffer[comma_index + 1 :]
    # yield the last chunk
    yield [buffer.strip()]
# %%
list_chain = str_chain | split_into_list
# %%
for chunk in list_chain.stream({"animal": "bear"}):
    print(chunk, flush=True)
# %%
list_chain.invoke({"animal": "bear"})
# %%
