from typing import Literal
from ollama import Client
from llama_index.llms.ollama import Ollama
import os
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.pydantic_v1 import BaseModel, Field

from fastapi import FastAPI

app = FastAPI()

BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://192.168.222.236:11434")


@app.get("/ollama")
def call_ollama(
    text: str,
    model: str = "llama2",
    timeout: float = 60.0,
    mode: Literal["llamaindex", "ollama"] = "ollama",
    base_url: str = BASE_URL,
) -> dict:
    if mode == "llamaindex":
        llm = Ollama(base_url=base_url, model=model, request_timeout=timeout)
        response = llm.complete(text)
    elif mode == "ollama":
        client = Client(host=base_url, timeout=timeout)
        response = client.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return response


DEFAULT_SYSTEM_TEMPLATE = """
现在有一批数据， 每一条数据都是分析师研报的标题，我希望根据标题的内容，把数据打上三种标签， 即对于该企业的态度（超预期 一般 低于预期）； 数据中存在以下几个问题： 第一条： 由于分析师普遍过度乐观或者不方便直接表述对于企业的看衰， 我希望在明显表述看好的标题才记作超预期，  隐含的描述不看好的即记作低于预期；  大部分的标题只是描述调研情况，无法反映看好或者不看好的态度；  第二条： 有的研报 是企业或者行业的综合性描述或一些例行公事的汇报， 不涉及具体企业的深度研究，这类研报不是某个企业的分析，标记为空值；  以上情景及要注意的问题你是否理解

你能共使用以下工具:

{tools}

你必须总是使用以下的工具之一并且只回复满足以下schema的JSON object:

{{
  "tool": <name of the selected tool>,
  "tool_input": <parameters for the selected tool, matching the tool's JSON schema>
}}
"""


class ClassifiedFinanceTitleWithReason(BaseModel):
    """根据所提供的金融分析师报告的标题，分类分析师对于企业的态度"""

    answer: str = Field(
        ...,
        description="根据标题的内容，把数据打上标签，即对于该企业的态度：超预期、一般、低于预期、空值",
    )
    justification: str = Field(
        ...,
        # description="给出为什么给出此标签的理由",
        description="给出详细条列判段过程，为什么给出此标签的理由",
    )


@app.get("/ollama_finance_classification")
def ollama_finance_title_classification(
    text: str,
    model: str = "llama2",
    timeout: int = 60,
    base_url: str = BASE_URL,
    mode: Literal["pydantic", "bind_tools"] = "pydantic",
    include_raw: bool = False,
):
    """
    https://github.com/langchain-ai/langchain/blob/cccc8fbe2fe59bde0846875f67aa046aeb1105a3/libs/experimental/langchain_experimental/llms/ollama_functions.py#L110

    example: https://chatgpt.com/share/888ac291-a92e-4155-b6b8-ed0d70205152
    """

    llm = OllamaFunctions(
        base_url=base_url,
        # Invalid model error:
        # ValueError: Ollama call failed with status code 500. Details: {"error":"llama runner process no longer running: -1 "}
        model=model,
        format="json",
        timeout=timeout,
        include_raw=include_raw,
        # keep_alive="5m",
    )

    llm.tool_system_prompt_template = DEFAULT_SYSTEM_TEMPLATE

    if mode == "pydantic":
        structured_llm = llm.with_structured_output(
            ClassifiedFinanceTitleWithReason, include_raw=include_raw
        )
        response = structured_llm.invoke(text)
    elif mode == "bind_tools":
        # llm = llm.bind_tools(
        #     tools=[
        #         {
        #             "name": "get_current_weather",
        #             "description": "Get the current weather in a given location",
        #             "parameters": {
        #                 "type": "object",
        #                 "properties": {
        #                     "location": {
        #                         "type": "string",
        #                         "description": "The city and state, " "e.g. San Francisco, CA",
        #                     },
        #                     "unit": {
        #                         "type": "string",
        #                         "enum": ["celsius", "fahrenheit"],
        #                     },
        #                 },
        #                 "required": ["location"],
        #             },
        #         }
        #     ],
        #     function_call={"name": "get_current_weather"},
        # )
        raise NotImplementedError("Have not implemented bind_tools mode.")
    else:
        raise ValueError(f"Invalid mode: {mode}")
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8777, reload=True)
