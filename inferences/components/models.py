import os
from enum import Enum
from dotenv import load_dotenv
from llama_index.llms.bedrock import Bedrock
from llama_index.llms.openai import OpenAI

load_dotenv()


class AWSModelEnum(str, Enum):
    TITAN_TEXT_EXPRESS_V1 = "amazon.titan-text-express-v1"
    CLAUDE_V2_1 = "anthropic.claude-v2:1"


class OpenAIModelEnum(str, Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    GPT_4o = "gpt-4o"


def get_bedrock_li_text_model(text_model=AWSModelEnum.CLAUDE_V2_1, max_tokens=512, temperature=0.1):
    llm = Bedrock(
        model=text_model,
        temperature=temperature,
        max_tokens=max_tokens,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
        region_name=os.getenv("REGION"),
    )
    return llm


def get_openai_model(model: OpenAIModelEnum = OpenAIModelEnum.GPT_3_5_TURBO, max_tokens=512, temperature=0.1):
    return OpenAI(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature
    )
