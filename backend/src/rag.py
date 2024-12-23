from click import prompt
from langchain import hub
from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

from src.qdrant import vector_store

model = ChatOllama(model="llama3.2", temperature=0)

prompt = hub.pull("rlm/rag-prompt")


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = model.invoke(messages)
    return {"answer": response.content}


def get_answer_and_docs(question: str):
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    result = graph.invoke({"question": question})
    return {"answer": result["answer"], "context": result["context"]}
