#%%

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

#%%
template='''\
You are a helpful assistant that \
translates {input_language} to {output_language}.\
'''
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
system_message_prompt

#%%
human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
human_message_prompt
# %%
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
chat_prompt

# %%
chat_prompt.input_variables
# %%
chat_prompt_value = chat_prompt.format_prompt(
    input_language="English", 
    output_language="French", 
    text="I love programming."
)
chat_prompt_value
# %%
chat_messages = chat_prompt_value.to_messages()
chat_messages
# %%



