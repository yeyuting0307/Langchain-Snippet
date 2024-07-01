#%%
import openai
import json
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.schema.output_parser import (
    StrOutputParser
)
from langchain.schema.runnable import (
    RunnablePassthrough, RunnableBranch, 
    RunnableParallel, RunnableLambda, RunnableConfig
)
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

from dotenv import load_dotenv
load_dotenv()

#%%

# ================= Default reply chain =================
default_chain = (
    {"question": RunnablePassthrough()}
    | ChatPromptTemplate.from_messages(
        [
            ("system", "You are a professional customer service representative."),
            ("human", "{question}")
        ]
    )
    | ChatOpenAI()
    | StrOutputParser()
)


#%%
# ================= Determin Intent with Openai function_call =================
function = {
    "name": "check_generate_image_intent",
    "description": '''Determin if the every question sentence has strong intent to generate image.
        If it has intendency to generate image, return True, otherwise return False''',
    "parameters": {
        "type": "object",
        "properties": {
            "is_generate_image": {
                "type": "boolean",
                "description": "If it has intendency to generate image, return True, otherwise return False ",
            },
        },
        "required": ["is_generate_image"],
    },
}

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "always use function_call"),
        ("human", "{question}")
    ]
)
model = ChatOpenAI().bind(
    function_call = {"name" : "check_generate_image_intent"},
    functions = [function],
    
)

get_json = JsonOutputFunctionsParser()

def parse_or_fix(text: str, config: RunnableConfig):
    fixing_chain = (
        ChatPromptTemplate.from_template(
            "Fix the following text:\n\n```text\n{input}\n```\nError: {error}"
            " Don't narrate, just respond with the fixed data."
        )
        | ChatOpenAI()
        | StrOutputParser()
    )
    for _ in range(3):
        try:
            return json.loads(text)
        except Exception as e:
            text = fixing_chain.invoke({"input": text, "error": e}, config)
    return "Failed to parse"
    
json_parser = RunnableLambda(parse_or_fix)

output_parser = RunnableLambda(lambda d: d.get("is_generate_image"))

is_gen_img = (
    {"question": RunnablePassthrough()}
    | prompt
    | model
    | get_json.with_fallbacks([json_parser])
    | output_parser
)


#%%
check_list = [
    '想像一隻兔子在月亮上飛翔',
    '我頭痛肚子動想請假',
    '超人大戰哥吉拉',
    'Cyberpunk風格的神奈川沖浪',
    '介紹一下你自己',
    'Tesla Model S Plaid價格',
    '羅東運動公園附近有什麼好吃的',
    '給我一張貓咪後空翻的圖片',
]
for text in check_list:
    out = is_gen_img.invoke(text)
    print(text, out)

#%%
# ================= Generate Image via Dall-E-3  =================
import openai
client = openai.OpenAI()

def gen_img_from_dalle3(prompt):
    img = client.images.generate(
        model="dall-e-3",
        prompt=f"{prompt}",
        n=1,
        style = "vivid",
        size="1024x1024"
    )
    return img.data[0].url

gen_img = RunnableLambda(gen_img_from_dalle3)

#%%
from operator import itemgetter
gen_img_chain = (
    {"prompt": RunnablePassthrough() | itemgetter('question')}
    | gen_img
)
img_url = gen_img_chain.invoke({"question" : "超人大戰哥吉拉"})
img_url

# %%
# ================= Combine the chains =================
img_gen_router = RunnableBranch(
    (lambda x: x.get('is_generate_image') == True, gen_img_chain),
    (lambda x: x.get('is_generate_image') == False, default_chain),
    default_chain
)

chain_flow = (
    {
        "question": RunnablePassthrough(), 
        "is_generate_image": is_gen_img
    }
    | img_gen_router
)

#%%
chain_flow.invoke("想像一隻兔子在月亮上飛翔")

#%%
