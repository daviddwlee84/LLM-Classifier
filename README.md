# LLM-Classifier

Use LLM do text classification task

## Getting Started

```bash
fastapi dev ./app.py
```

## Solutions

- [OpenAI JSON Mode vs. Function Calling for Data Extraction - LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/llm/openai_json_vs_function_calling/)

### JSON Mode

- [JSON Mode - Text generation - OpenAI API](https://platform.openai.com/docs/guides/text-generation/json-mode)
  - JSON mode will not guarantee the output matches any specific schema, only that it is valid and parses without errors.

### Functional Calling

- [Function calling - OpenAI API](https://platform.openai.com/docs/guides/function-calling)
  - Function calling allows you to more reliably get structured data back from the model
- [How to call functions with chat models | OpenAI Cookbook](https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models)

### LlamaIndex Pydantic Program

- [Pydantic Program - LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/querying/structured_outputs/pydantic_program/)
- [OpenAI Pydantic Program - LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/output_parsing/openai_pydantic_program/)
  - This guide shows you how to generate structured data with new OpenAI API ([Function calling and other API updates | OpenAI](https://openai.com/index/function-calling-and-other-api-updates/)) via LlamaIndex. The user just needs to specify a Pydantic object.

### Output Parsing

- [Langchain Output Parsing - LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/output_parsing/LangchainOutputParserDemo/)

## Resources

- [Ollama - LlamaIndex](https://docs.llamaindex.ai/en/stable/api_reference/llms/ollama/)

### Model

- [Unleashing LLMs: Functional Calling with LangChain, Ollama, and Microsoft’s Phi-3 (PART-2): | by Anoop Maurya | May, 2024 | Medium](https://medium.com/@mauryaanoop3/unleashing-llms-functional-calling-with-langchain-ollama-and-microsofts-phi-3-part-2-10fae91d7b01)

Functional Calling Model

- [nexusraven](https://ollama.com/library/nexusraven)
- [smangrul/llama-3-8b-instruct-function-calling](https://ollama.com/smangrul/llama-3-8b-instruct-function-calling)
- [phi3](https://ollama.com/library/phi3) (no explicitly said support functional calling)
