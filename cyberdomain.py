import streamlit as st
from langchain_openai import OpenAIEmbeddings
import time
from pinecone import Pinecone
import numpy as np
import sys
from abc import ABC, abstractmethod
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI
import concurrent.futures
import structlog

st.title("Defendify: Your Cybersecurity Assistant")

# Prompt user to input API keys and their cybersecurity-related question
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
PINECONE_API_KEY = st.secrets["general"]["PINECONE_API_KEY"]

# Prompt user to input their cybersecurity question
cybersecurity_question = st.text_area("What would you like to ask about cybersecurity? ")

PINECONE_INDEX = "cyberdomain"
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

class BasePromptTemplate(ABC, BaseModel):
    @abstractmethod
    def create_template(self, *args) -> PromptTemplate:
        pass


class QueryExpansionTemplate(BasePromptTemplate):
    prompt: str = """You are an AI language model assistant. Your task is to generate {to_expand_to_n}
    different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions separated by '{separator}'.
    Original question: {question}"""

    @property
    def separator(self) -> str:
        return "#next-question#"

    def create_template(self, to_expand_to_n: int) -> PromptTemplate:
        return PromptTemplate(
            template=self.prompt,
            input_variables=["question"],
            partial_variables={
                "separator": self.separator,
                "to_expand_to_n": to_expand_to_n,
            },
        )


class SelfQueryTemplate(BasePromptTemplate):
    prompt: str = """You are an AI language model assistant. Your task is to extract information from a user question.
    The required information that needs to be extracted is the user or author id. 
    Your response should consist of only the extracted id (e.g. 1345256), nothing else.
    User question: {question}"""

    def create_template(self) -> PromptTemplate:
        return PromptTemplate(template=self.prompt, input_variables=["question"])


class RerankingTemplate(BasePromptTemplate):
    prompt: str = """You are an AI language model assistant. Your task is to rerank passages related to a query
    based on their relevance. 
    You should only return the summary of the most relevant passage.
    
    The following are passages related to this query: {question}.
    
    Passages: 
    {passages}
    
    Please provide only the summary of the most relevant passage.
    """

    def create_template(self, keep_top_k: int) -> PromptTemplate:
        return PromptTemplate(
            template=self.prompt,
            input_variables=["question", "passages"],
            partial_variables={"keep_top_k": keep_top_k, "separator": self.separator},
        )

    @property
    def separator(self) -> str:
        return "\n#next-document#\n"


class GeneralChain:
    @staticmethod
    def get_chain(llm, template: PromptTemplate, output_key: str, verbose=True):
        return LLMChain(
            llm=llm, prompt=template, output_key=output_key, verbose=verbose
        )


class QueryExpansion:
    @staticmethod
    def generate_response(query: str, to_expand_to_n: int) -> list[str]:
        query_expansion_template = QueryExpansionTemplate()
        prompt_template = query_expansion_template.create_template(to_expand_to_n)
        model = ChatOpenAI(
            model="gpt-4-1106-preview",
            api_key=openai_api_key,
            temperature=0,
        )

        chain = GeneralChain().get_chain(
            llm=model, output_key="expanded_queries", template=prompt_template
        )

        response = chain.invoke({"question": query})
        result = response["expanded_queries"]

        queries = result.strip().split(query_expansion_template.separator)
        stripped_queries = [
            stripped_item for item in queries if (stripped_item := item.strip())
        ]

        return stripped_queries


class SelfQuery:
    @staticmethod
    def generate_response(query: str) -> str:
        prompt = SelfQueryTemplate().create_template()
        model = ChatOpenAI(
            model="gpt-4-1106-preview",
            api_key=openai_api_key,
            temperature=0,
        )

        chain = GeneralChain().get_chain(
            llm=model, output_key="metadata_filter_value", template=prompt
        )

        response = chain.invoke({"question": query})
        result = response["metadata_filter_value"]

        return result


class Reranker:
    @staticmethod
    def generate_response(
        query: str, passages: list[str], keep_top_k: int
    ) -> list[str]:
        reranking_template = RerankingTemplate()
        prompt_template = reranking_template.create_template(keep_top_k=keep_top_k)

        model = ChatOpenAI(
            model="gpt-4-1106-preview",
            api_key=openai_api_key,
            temperature=0,
        )
        chain = GeneralChain().get_chain(
            llm=model, output_key="rerank", template=prompt_template
        )

        stripped_passages = [
            stripped_item for item in passages if (stripped_item := item.strip())
        ]
        passages = reranking_template.separator.join(stripped_passages)
        response = chain.invoke({"question": query, "passages": passages})

        result = response["rerank"]
        reranked_passages = result.strip().split(reranking_template.separator)
        stripped_passages = [
            stripped_item
            for item in reranked_passages
            if (stripped_item := item.strip())
        ]

        return stripped_passages


def get_logger(cls: str):
    return structlog.get_logger().bind(cls=cls)

logger = get_logger(__name__)

def flatten(nested_list: list) -> list:
    """Flatten a list of lists into a single list."""
    return [item for sublist in nested_list for item in sublist]


class VectorRetriever:
    """
    Class for retrieving vectors from a Vector store in a RAG system using query expansion and Multitenancy search.
    """

    def __init__(self, index, query: str) -> None:
        self._client = index
        self.query = query
        self._embedder = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self._query_expander = QueryExpansion()
        self._metadata_extractor = SelfQuery()
        self._reranker = Reranker()

    def _search_single_query(
        self, generated_query: str, k: int, include_metadata=True
        ):
        assert k > 3, "k should be greater than 3"

        query_vector = self._embedder.embed_query(generated_query)

        # Query Pinecone using the embedded vector
        query_results = self._client.query(
            vector=query_vector,  # Correct placement for the vector
            top_k=k // 3,
            include_metadata=True
        )

        return query_results['matches']


    def retrieve_top_k(self, k: int, to_expand_to_n_queries: int) -> list:
        generated_queries = self._query_expander.generate_response(
            self.query, to_expand_to_n=to_expand_to_n_queries
        )

        logger.info("Successfully generated queries for search.", num_queries=len(generated_queries))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            search_tasks = [
                executor.submit(self._search_single_query, query, k, include_metadata=True)
                for query in generated_queries
            ]

            hits = [task.result() for task in concurrent.futures.as_completed(search_tasks)]

            # Since 'hits' is a list of lists (matches), flatten them manually
            hits = [item for sublist in hits for item in sublist]

        logger.info("All documents retrieved successfully.", num_documents=len(hits))

        return hits


    def rerank(self, hits: list, keep_top_k: int) -> str:
        # Extract the 'text' field from 'metadata' for each hit
        content_list = [hit['metadata']['text'] for hit in hits if hit and hit.get('metadata') and hit['metadata'].get('text')]

        rerank_hits = self._reranker.generate_response(
            query=self.query, passages=content_list, keep_top_k=keep_top_k
        )

        # Return the first reranked hit as the best answer
        if rerank_hits:
            best_answer = rerank_hits[0]
            logger.info(f"Best answer selected: {best_answer}")
            return best_answer
        else:
            return ""


    def set_query(self, query: str):
        self.query = query


# Submit button
if st.button("Submit"):
    # Check if API keys are provided and a question is entered
    if openai_api_key and PINECONE_API_KEY and cybersecurity_question:
        # Initialize the retriever with the index and question
        retriever = VectorRetriever(index=index, query=cybersecurity_question)

        # Define parameters for retrieval
        k = 10  # Number of results to retrieve
        to_expand_to_n_queries = 5  # Number of expanded queries to generate

        # Perform the vector search to retrieve the top-k documents
        top_k_hits = retriever.retrieve_top_k(k=k, to_expand_to_n_queries=to_expand_to_n_queries)

        # Optional: Rerank the retrieved hits and get the best answer (the top result)
        best_answer = retriever.rerank(hits=top_k_hits, keep_top_k=1)  # keep only the top result

        # Display the best answer in the app
        st.write(f"Best answer: {best_answer}")
    else:
        st.warning("Please provide both API keys and ask a question.")
