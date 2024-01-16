import os
from openai import AzureOpenAI

from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings


def get_completions_gpt35_16k(msg, temp=0.1):
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_GPT35_16K_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_GPT35_16K_ENDPOINT"),
    )
    completion = client.chat.completions.create(
        model=os.getenv("gpt-35-turbo-16k"), messages=msg, temperature=temp
    )

    return completion


def get_completions_gpt4(msg, temp=0.1):
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_GPT4_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_GPT4_ENDPOINT"),
    )
    completion = client.chat.completions.create(
        model="chat-gpt-4", messages=msg, temperature=temp
    )

    return completion

def get_completions_gpt4_turbo(msg, temp=0.1):
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_GPT4TURBO_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_GPT4TURBO_ENDPOINT"),
    )
    completion = client.chat.completions.create(
        model="gpt-4-turbo", response_format={"type": "json_object"}, messages=msg, temperature=temp
    )

    return completion

def get_completions_gpt35(msg, temp=0.1):
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_GPT35_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_GPT35_ENDPOINT"),
    )
    completion = client.chat.completions.create(
        model="chat", messages=msg, temperature=temp
    )

    return completion


def get_embeddings_ada(input):
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_ADA_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ADA_ENDPOINT"),
    )
    response = client.embeddings.create(input=input, model="text-embedding-ada-002")

    return response.data[0].embedding


def get_langchain_gpt35():
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
        model="gpt-3.5-turbo-16k",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )


def get_langchain_gpt4():
    return AzureChatOpenAI(
        azure_deployment="chat-gpt-4",
        model="gpt-4",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )


def get_langchain_embed_model():
    return AzureOpenAIEmbeddings(
        model="text-embedding-ada-002",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
