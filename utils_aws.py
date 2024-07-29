import json
import os
from enum import Enum
from pprint import pp

import boto3
from dotenv import load_dotenv

load_dotenv()


class BedrockModelEnum(str, Enum):
    TITAN_TEXT_EXPRESS_V1 = "amazon.titan-text-express-v1"
    CLAUDE_V2_1 = "anthropic.claude-v2:1"


class BedrockEmbeddingEnum(str, Enum):
    TITAN_EMBED_TEXT_V1 = "amazon.titan-embed-text-v1"
    COHERE_EMBED_ENGLISH_V3 = "cohere.embed-english-v3"
    COHERE_EMBED_MULTILINGUAL_V3 = "cohere.embed-multilingual-v3"


class TextException(Exception):
    def __init__(self, message):
        self.message = message


class BedrockTextModelBotoUtils:
    def __init__(self, model_id=BedrockModelEnum.CLAUDE_V2_1):
        self.model_id = model_id
        self.bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name=os.environ["REGION"],
            aws_access_key_id=os.environ["AWS_ACCESS_KEY"],
            aws_secret_access_key=os.environ["AWS_SECRET_KEY"]
        )

    def generate_text(self, body: dict):
        response = self.bedrock_runtime.invoke_model(
            body=json.dumps(body),
            modelId=self.model_id,
            accept="application/json",
            contentType="application/json"
        )

        response_body = json.loads(response.get("body").read())
        finish_reason = response_body.get("error")
        if finish_reason is not None:
            raise TextException(f"Text generation error. Error is {finish_reason}")

        return response_body

    def test_call(self, input_text: str = "Human: Who are you\nAssistant:"):
        if self.model_id == BedrockModelEnum.CLAUDE_V2_1:
            body = {
                "prompt": input_text,
                "max_tokens_to_sample": 200,
                "temperature": 0.5,
                "stop_sequences": ["\n\nHuman:"],
            }
            response_body = self.generate_text(body)
            completion = response_body["completion"]
            return completion

        elif self.model_id == BedrockModelEnum.TITAN_TEXT_EXPRESS_V1:
            body = {
                "inputText": input_text,
                "textGenerationConfig": {
                    "maxTokenCount": 3072,
                    "stopSequences": [],
                    "temperature": 0.9,
                    "topP": 0.9
                }
            }
            response_body = self.generate_text(body)
            print(f"Input token count: {response_body['inputTextTokenCount']}")
            for result in response_body['results']:
                print(f"Token count: {result['tokenCount']}")
                print(f"Output text: {result['outputText']}")
                print(f"Completion reason: {result['completionReason']}")
            return response_body['results'][0]['outputText']


def get_aws_foundation_model_list():
    bedrock = boto3.client(
        service_name="bedrock",
        region_name=os.environ["REGION"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY"],
        aws_secret_access_key=os.environ["AWS_SECRET_KEY"]
    )

    pp(bedrock.list_foundation_models())


class BedrockEmbedModelBotoUtils:
    def __init__(self, model_id=BedrockEmbeddingEnum.TITAN_EMBED_TEXT_V1.value):
        self.model_id = model_id
        self.bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name=os.environ["REGION"],
            aws_access_key_id=os.environ["AWS_ACCESS_KEY"],
            aws_secret_access_key=os.environ["AWS_SECRET_KEY"]
        )

    def generate_embedding(self, body):
        response = self.bedrock_runtime.invoke_model(
            body=body,
            modelId=self.model_id,
            accept="application/json",
            contentType="application/json"
        )

        response_body = json.loads(response.get('body').read())

        return response_body

    def test_call(self, input_text="Retrieve random embeddings"):
        # TODO: branching cohere embeddings body params
        body = json.dumps({
            "inputText": input_text,
        })

        response = self.generate_embedding(body)

        print(f"Generated embeddings: {response['embedding']}")
        print(f"Input Token count:  {response['inputTextTokenCount']}")
