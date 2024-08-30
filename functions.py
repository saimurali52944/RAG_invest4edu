from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from langchain_community.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pymongo import MongoClient
from datetime import datetime
import config

class SearchApp:
    def __init__(self):
        def get_openai_client():
            return AzureOpenAI(
                api_key=config.AZURE_OPENAI_API_KEY,
                api_version=config.AZURE_OPENAI_API_VERSION,
                azure_endpoint=config.AZURE_OPENAI_ENDPOINT
            )

        def get_search_client():
            return SearchClient(
                config.AZURE_SEARCH_ENDPOINT,
                config.AZURE_SEARCH_INDEX_NAME,
                AzureKeyCredential(config.AZURE_SEARCH_KEY)
            )

        def get_mongo_client():
            client = MongoClient(config.MONGO_CONNECTION_STRING)
            db = client[config.MONGO_DATABASE_NAME]
            return db[config.MONGO_COLLECTION_NAME]

        def get_chat_model():
            return AzureChatOpenAI(
                azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
                openai_api_version=config.AZURE_OPENAI_API_VERSION,
                deployment_name=config.AZURE_OPENAI_DEPLOYMENT_NAME,
                openai_api_key=config.AZURE_OPENAI_API_KEY,
                openai_api_type="azure",
                max_tokens=400,
                frequency_penalty=1,
                temperature=0.4
            )

        def get_embeddings(text):
            client = get_openai_client()
            response = client.embeddings.create(
                input=text,
                model=config.AZURE_OPENAI_EMBEDDING_MODEL
            )
            return response.data[0].embedding

        def perform_vector_search(query, k=5):
            search_client = get_search_client()
            query_vector = get_embeddings(query)
            vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=k, fields="vector")
            
            results = search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                select=["chunk_id", "parent_id", "chunk", "title"]
            )
            
            return list(results)

        def get_llm_chain():
            model = get_chat_model()
            template = """
            You are an AI assistant specializing in educational policies. Provide the summary of the information in an elaborate manner.
            Follow Up Input:
            question: {Question}
            docs: {docs}
            """
            prompt = PromptTemplate(input_variables=["Question", "docs"], template=template)
            return LLMChain(llm=model, prompt=prompt, verbose=True)

        def log_search(user_id, query, output):
            collection = get_mongo_client()
            log_entry = {
                "user_id": user_id,
                "query": query,
                "output": output,
                "timestamp": datetime.utcnow()
            }
            collection.insert_one(log_entry)

        self.get_openai_client = get_openai_client
        self.get_search_client = get_search_client
        self.get_mongo_client = get_mongo_client
        self.get_chat_model = get_chat_model
        self.get_embeddings = get_embeddings
        self.perform_vector_search = perform_vector_search
        self.get_llm_chain = get_llm_chain
        self.log_search = log_search