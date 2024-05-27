from typing import Literal
from ollama import Client
from llama_index.llms.ollama import Ollama
import os
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.utils.json import (
    parse_and_check_json_markdown,
    parse_json_markdown,
    parse_partial_json,
)

from fastapi import FastAPI

app = FastAPI()

BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://192.168.222.236:11434")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")


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
    model: Literal[
        "llama3", "phi3"
    ] = "llama3",  # currently, phi3 are not usable; llama2 doesn't support function calling
    timeout: int = 60,
    base_url: str = BASE_URL,
    mode: Literal[
        "pydantic", "bind_tools", "json_parser", "raw_json_parser", "v2_raw_json_parser"
    ] = "v2_raw_json_parser",
    include_raw: bool = False,
    force_json_mode: bool = True,
):
    """
    https://github.com/langchain-ai/langchain/blob/cccc8fbe2fe59bde0846875f67aa046aeb1105a3/libs/experimental/langchain_experimental/llms/ollama_functions.py#L110

    example: https://chatgpt.com/share/888ac291-a92e-4155-b6b8-ed0d70205152

    https://github.com/run-llama/llama_index/issues/7587

    https://github.com/ollama/ollama/blob/main/docs/api.md#json-mode
    """

    if not mode.endswith("json_parser"):
        # https://github.com/langchain-ai/langchain/issues/20513
        llm = OllamaFunctions(
            base_url=base_url,
            # Invalid model error:
            # ValueError: Ollama call failed with status code 500. Details: {"error":"llama runner process no longer running: -1 "}
            model=model,
            # ValueError: 'llama3' did not respond with valid JSON.
            # format=None if mode.endswith("json_parser") else "json",
            # format="json",
            format="json" if force_json_mode else None,
            timeout=timeout,
            include_raw=include_raw,
            # keep_alive="5m",
        )

        llm.tool_system_prompt_template = DEFAULT_SYSTEM_TEMPLATE
    else:
        # https://python.langchain.com/v0.1/docs/integrations/chat/ollama/
        llm = ChatOllama(
            base_url=base_url,
            model=model,
            # format="json" if mode == 'v2_raw_json_parser' else None, # This will become super slow and get weird result
            format="json" if force_json_mode else None,
            timeout=timeout,
        )

    if mode == "pydantic":
        structured_llm = llm.with_structured_output(
            ClassifiedFinanceTitleWithReason, include_raw=include_raw
        )
        # langchain_core.exceptions.OutputParserException: Failed to parse ClassifiedFinanceTitleWithReason from completion {"title": "\u8d85\u9884\u671f", "reason": "\u660e\u663e\u8868\u8ff0\u770b\u597d\u7684\u6807\u9898"}. Got: 2 validation errors for ClassifiedFinanceTitleWithReason
        response = structured_llm.invoke(text)
    elif mode == "bind_tools":
        llm = llm.bind_tools(
            tools=[
                {
                    "name": "finance_title_classification",
                    "description": "根据所提供的金融分析师报告的标题，分类分析师对于企业的态度",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "string",
                                "enum": ["超预期", "一般", "低于预期", "空值"],
                                "description": "根据标题的内容，把数据打上三种标签， 即对于该企业的态度（超预期 一般 低于预期）； 数据中存在以下几个问题： 第一条： 由于分析师普遍过度乐观或者不方便直接表述对于企业的看衰， 我希望在明显表述看好的标题才记作超预期，  隐含的描述不看好的即记作低于预期；  大部分的标题只是描述调研情况，无法反映看好或者不看好的态度；  第二条： 有的研报 是企业或者行业的综合性描述或一些例行公事的汇报， 不涉及具体企业的深度研究，这类研报不是某个企业的分析，标记为空值；",
                            },
                            "justification": {
                                "type": "string",
                                "description": "详细条列判段过程，为什么给出此标签的理由",
                            },
                        },
                        # "required": ["公司或企业名称"],
                        "required": ["answer", "justification"],
                    },
                }
            ],
            function_call={"name": "finance_title_classification"},
        )
        # KeyError: 'tool'
        response = llm.invoke(text)
    elif mode == "json_parser":
        parser = JsonOutputParser(pydantic_object=ClassifiedFinanceTitleWithReason)
        # BUG: Chinese character formatting issue
        # As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
        # the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.
        #
        # Here is the output schema:
        # ```
        # {"description": "\u6839\u636e\u6240\u63d0\u4f9b\u7684\u91d1\u878d\u5206\u6790\u5e08\u62a5\u544a\u7684\u6807\u9898\uff0c\u5206\u7c7b\u5206\u6790\u5e08\u5bf9\u4e8e\u4f01\u4e1a\u7684\u6001\u5ea6", "properties": {"answer": {"title": "Answer", "description": "\u6839\u636e\u6807\u9898\u7684\u5185\u5bb9\uff0c\u628a\u6570\u636e\u6253\u4e0a\u6807\u7b7e\uff0c\u5373\u5bf9\u4e8e\u8be5\u4f01\u4e1a\u7684\u6001\u5ea6\uff1a\u8d85\u9884\u671f\u3001\u4e00\u822c\u3001\u4f4e\u4e8e\u9884\u671f\u3001\u7a7a\u503c", "type": "string"}, "justification": {"title": "Justification", "description": "\u7ed9\u51fa\u8be6\u7ec6\u6761\u5217\u5224\u6bb5\u8fc7\u7a0b\uff0c\u4e3a\u4ec0\u4e48\u7ed9\u51fa\u6b64\u6807\u7b7e\u7684\u7406\u7531", "type": "string"}}, "required": ["answer", "justification"]}
        # ```
        print(parser.get_format_instructions())
        prompt = PromptTemplate(
            template="现在有一批数据， 每一条数据都是分析师研报的标题，我希望根据标题的内容，把数据打上三种标签， 即对于该企业的态度（超预期 一般 低于预期）； 数据中存在以下几个问题： 第一条： 由于分析师普遍过度乐观或者不方便直接表述对于企业的看衰， 我希望在明显表述看好的标题才记作超预期，  隐含的描述不看好的即记作低于预期；  大部分的标题只是描述调研情况，无法反映看好或者不看好的态度；  第二条： 有的研报 是企业或者行业的综合性描述或一些例行公事的汇报， 不涉及具体企业的深度研究，这类研报不是某个企业的分析，标记为空值；\n{format_instructions}\n{title}",
            input_variables=["title"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | llm | parser
        response = chain.invoke({"title": text})

    elif mode == "raw_json_parser":
        format_instructions = """
        回答需要符合以下JSON格式

        举例来说
        对于格式 `{"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}` 来说
        `{"foo": ["bar", "baz"]}` 就是一个符合标准的回答. 而 `{"properties": {"foo": ["bar", "baz"]}}` 则不符合标准.

        以下是输出格式规范
        ```
        {
            "properties":{
                "answer": {
                    "type": "string",
                    "enum": ["超预期", "一般", "低于预期", "空值"],
                    "description": "根据标题的内容，把数据打上标签，即对于该企业的态度：超预期、一般、低于预期、空值",
                },
                "justification": {
                    "type": "string",
                    "description": "详细条列判段过程，为什么给出此标签的理由",
                },
            },
            "required": ["answer", "justification"]
        }
        ```
        """
        prompt = PromptTemplate(
            template="现在有一批数据， 每一条数据都是分析师研报的标题，我希望根据标题的内容，把数据打上三种标签， 即对于该企业的态度（超预期 一般 低于预期）； 数据中存在以下几个问题： 第一条： 由于分析师普遍过度乐观或者不方便直接表述对于企业的看衰， 我希望在明显表述看好的标题才记作超预期，  隐含的描述不看好的即记作低于预期；  大部分的标题只是描述调研情况，无法反映看好或者不看好的态度；  第二条： 有的研报 是企业或者行业的综合性描述或一些例行公事的汇报， 不涉及具体企业的深度研究，这类研报不是某个企业的分析，标记为空值；\n{format_instructions}\n{title}",
            input_variables=["title"],
            partial_variables={"format_instructions": format_instructions},
        )
        chain = prompt | llm | StrOutputParser()
        raw_response = chain.invoke({"title": text})
        try:
            response = parse_and_check_json_markdown(
                # raw_response, expected_keys=["answer", "justification"]
                raw_response,
                expected_keys=["answer"],
            )
        except:
            response = raw_response
    elif "v2_raw_json_parser":
        format_instructions = """
        回答需要符合以下JSON格式
        ```json
        {
            "answer": "根据标题的内容，把数据打上标签，即对于该企业的态度：超预期、一般、低于预期、空值"
            "justification": "详细条列判段过程，为什么给出此标签的理由"
        }
        ```

        例如
        title: 喜临门：“家具下乡”提振需求，股权激励赋能发展
        json:
        {
            "answer": "超预期"
            "justification": "1. 提振需求：标题中提到“家具下乡”政策提振需求，这是一个积极的市场信号，暗示企业可能会从中受益。 2. 股权激励赋能发展：股权激励通常被视为企业激励员工、推动企业长期发展的重要措施，暗示管理层对未来发展有信心。 3. 整体语气：整体语气积极，显然是在表达看好企业未来发展。"
        }
        title: 伟星新材系列深度报告之三：对标海天，伟星零售护城河优势尽显
        json:
        {
            "answer": "一般"
            "justification": "1. 系列深度报告：标题中提到这是系列深度报告的第三篇，表明这是对企业的全面分析的一部分，但并不直接表明态度。 2. 对标海天：提到对标另一家企业（海天），但未明确表达出看好或不看好的态度。 3. 优势尽显：虽然提到了“零售护城河优势尽显”，这是一个积极的描述，但没有使用非常强烈的表述来表明超预期的态度。"
        }
        title: 于模糊中寻找确定性——2022年食品饮料行业年度策略
        json:
        {
            "answer": "空值",
            "justification": "1. 行业年度策略：标题明确指出这是对整个食品饮料行业的年度策略分析，而不是针对某个具体企业的分析。 2. 于模糊中寻找确定性：这是一个较为笼统的描述，侧重于行业的整体策略和趋势，而非具体企业的表现或预期。 3. 缺乏具体企业信息：标题中没有提及任何具体企业，因此无法判断对某一企业的态度。"
        }
        """
        # "justification": "1. 该标题并没有涉及具体企业的深度研究，而是对某个行业或行业的整体描述，无法反映对于特定企业的态度。 2. 标题中没有明确表达出看好或不看好的态度，也没有使用强烈的语言来表明超预期的态度。"
        prompt = PromptTemplate(
            template="现在有一批数据， 每一条数据都是分析师研报的标题，我希望根据标题的内容，把数据打上三种标签， 即对于该企业的态度（超预期 一般 低于预期）； 数据中存在以下几个问题： 第一条： 由于分析师普遍过度乐观或者不方便直接表述对于企业的看衰， 我希望在明显表述看好的标题才记作超预期，  隐含的描述不看好的即记作低于预期；  大部分的标题只是描述调研情况，无法反映看好或者不看好的态度；  第二条： 有的研报 是企业或者行业的综合性描述或一些例行公事的汇报， 不涉及具体企业的深度研究，这类研报不是某个企业的分析，标记为空值；\n{format_instructions}\ntitle: {title}\njson:\n",
            input_variables=["title"],
            partial_variables={"format_instructions": format_instructions},
        )
        chain = prompt | llm | StrOutputParser()
        raw_response = chain.invoke({"title": text})
        try:
            # NOTE: basically only works in json mode
            response = parse_and_check_json_markdown(
                # raw_response, expected_keys=["answer", "justification"]
                raw_response,
                expected_keys=["answer"],
            )
        except:
            response = raw_response
    else:
        raise ValueError(f"Invalid mode: {mode}")
    return response


@app.get("/openai_finance_classification")
def openai_finance_title_classification(
    text: str,
    model: Literal["gpt-3.5-turbo"] = "gpt-3.5-turbo",
    timeout: int = 60,
    api_key: str = OPENAI_API_KEY,
    mode: Literal["v2_raw_json_parser"] = "v2_raw_json_parser",
    force_json_mode: bool = True,
):
    if not mode.endswith("json_parser"):
        raise NotImplementedError()
    else:
        llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            timeout=timeout,
        )
        if force_json_mode:
            llm = llm.bind(response_format={"type": "json_object"})

    if "v2_raw_json_parser":
        format_instructions = """
        回答需要符合以下JSON格式
        ```json
        {
            "answer": "根据标题的内容，把数据打上标签，即对于该企业的态度：超预期、一般、低于预期、空值"
            "justification": "详细条列判段过程，为什么给出此标签的理由"
        }
        ```

        例如
        title: 喜临门：“家具下乡”提振需求，股权激励赋能发展
        json:
        {
            "answer": "超预期"
            "justification": "1. 提振需求：标题中提到“家具下乡”政策提振需求，这是一个积极的市场信号，暗示企业可能会从中受益。 2. 股权激励赋能发展：股权激励通常被视为企业激励员工、推动企业长期发展的重要措施，暗示管理层对未来发展有信心。 3. 整体语气：整体语气积极，显然是在表达看好企业未来发展。"
        }
        title: 伟星新材系列深度报告之三：对标海天，伟星零售护城河优势尽显
        json:
        {
            "answer": "一般"
            "justification": "1. 系列深度报告：标题中提到这是系列深度报告的第三篇，表明这是对企业的全面分析的一部分，但并不直接表明态度。 2. 对标海天：提到对标另一家企业（海天），但未明确表达出看好或不看好的态度。 3. 优势尽显：虽然提到了“零售护城河优势尽显”，这是一个积极的描述，但没有使用非常强烈的表述来表明超预期的态度。"
        }
        title: 于模糊中寻找确定性——2022年食品饮料行业年度策略
        json:
        {
            "answer": "空值",
            "justification": "1. 行业年度策略：标题明确指出这是对整个食品饮料行业的年度策略分析，而不是针对某个具体企业的分析。 2. 于模糊中寻找确定性：这是一个较为笼统的描述，侧重于行业的整体策略和趋势，而非具体企业的表现或预期。 3. 缺乏具体企业信息：标题中没有提及任何具体企业，因此无法判断对某一企业的态度。"
        }
        """
        # "justification": "1. 该标题并没有涉及具体企业的深度研究，而是对某个行业或行业的整体描述，无法反映对于特定企业的态度。 2. 标题中没有明确表达出看好或不看好的态度，也没有使用强烈的语言来表明超预期的态度。"
        prompt = PromptTemplate(
            template="现在有一批数据， 每一条数据都是分析师研报的标题，我希望根据标题的内容，把数据打上三种标签， 即对于该企业的态度（超预期 一般 低于预期）； 数据中存在以下几个问题： 第一条： 由于分析师普遍过度乐观或者不方便直接表述对于企业的看衰， 我希望在明显表述看好的标题才记作超预期，  隐含的描述不看好的即记作低于预期；  大部分的标题只是描述调研情况，无法反映看好或者不看好的态度；  第二条： 有的研报 是企业或者行业的综合性描述或一些例行公事的汇报， 不涉及具体企业的深度研究，这类研报不是某个企业的分析，标记为空值；\n{format_instructions}\ntitle: {title}\njson:\n",
            input_variables=["title"],
            partial_variables={"format_instructions": format_instructions},
        )
        chain = prompt | llm | StrOutputParser()
        raw_response = chain.invoke({"title": text})
        try:
            # NOTE: basically only works in json mode
            response = parse_and_check_json_markdown(
                # raw_response, expected_keys=["answer", "justification"]
                raw_response,
                expected_keys=["answer"],
            )
        except:
            response = raw_response
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8777, reload=True)
