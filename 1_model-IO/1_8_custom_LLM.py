#%%
import os
from dotenv import load_dotenv
load_dotenv('../.env') # OPENAI_API_KEY

#%%
from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

import openai

# %%
class CustomLLM(LLM):
    model_name = "gpt-3.5-turbo"
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        answer = ""
        for wp in self.StreamChatGPT(prompt): #TODO: streaming failed
            answer += wp
        return answer

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_name": self.model_name}
    
    def StreamChatGPT(self, question, model = "gpt-3.5-turbo", temperature = 0, max_tokens = 2000):
        messages = [{"role": "user", "content": f"{question}"}]
        completion = openai.ChatCompletion.create(
            model = self.model_name or model,
            messages = messages,
            temperature = temperature,
            max_tokens = max_tokens,
            stream = True
        )
        for wp in completion:
            content = wp.choices[0].delta.get("content")
            finish = wp.choices[0].finish_reason
            if content:
                yield content
            elif finish == "stop":
                break
            else:
                continue
# %%
llm = CustomLLM(model_name = 'gpt-3.5-turbo-16k')
# %%
llm("用50個字形容深度學習領域的Embedding是什麼")
# %%
