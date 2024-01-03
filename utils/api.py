import os
from openai import AzureOpenAI

from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings

client = AzureOpenAI(
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)


def get_completions_gpt35_16k(msg, temp=0.1):
    completion = client.chat.completions.create(
        model=os.getenv("AZURE_DEPLOYMENT_NAME"), messages=msg, temperature=temp
    )

    return completion


def get_completions_gpt4(msg, temp=0.1):
    completion = client.chat.completions.create(
        model="chat-gpt-4", messages=msg, temperature=temp
    )

    return completion


def get_completions_gpt35(msg, temp=0.1):
    completion = client.chat.completions.create(
        model="chat", messages=msg, temperature=temp
    )

    return completion


def get_embeddings_ada(input):
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
