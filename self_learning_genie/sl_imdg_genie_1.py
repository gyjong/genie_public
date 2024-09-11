import os
import streamlit as st
import datetime
import uuid
import logging
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from langchain.schema import HumanMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START, END, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Dict
from typing_extensions import TypedDict
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.schema import AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_community.graphs import Neo4jGraph
from langchain.agents.agent_types import AgentType
from neo4j import GraphDatabase
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# LangSmith 
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "imdg-self-learning-agent"

# Initialize API keys and connections
neo4j_url = os.getenv("NEO4J_URL")
neo4j_user = os.getenv("NEO4J_USER")
neo4j_password = os.getenv("NEO4J_PASSWORD")

# Initialize Neo4j connection
graph = Neo4jGraph(
    url=neo4j_url, 
    username=neo4j_user, 
    password=neo4j_password
)
driver = GraphDatabase.driver(
    neo4j_url, 
    auth=(neo4j_user, neo4j_password)
)

# Initialize Groq LLM
# llm = ChatGroq(model="llama-3.1-405b-reasoning", temperature=0.0)
# llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.0)
# llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
# llm = ChatGroq(model="gemma2-9b-it", temperature=0.0)
# llm = ChatGroq(model="llama3-70b-8192", temperature=0.0)
# llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)
# llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=1.0)

# Apply custom font
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Avenir:wght@400;700&display=swap');
    html, body, [class*="st-"] {
        font-family: 'Avenir', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to load documents
def load_documents(sources: List[str]) -> List[Document]:
    docs = []
    for source in sources:
        if source.startswith('http'):
            loader = WebBaseLoader(source)
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata['source'] = source
            docs.extend(loaded_docs)
        elif source.endswith('.pdf'):
            loader = PyPDFLoader(source)
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata['source'] = f"{source} (Page {doc.metadata['page']})"
            docs.extend(loaded_docs)
        else:
            raise ValueError(f"Unsupported source type: {source}")
    return docs

# Function to create or load vector store
def create_or_load_vectorstore(sources: List[str], index_name: str = "index_imdg") -> FAISS:
    current_dir = os.getcwd()
    index_dir = os.path.join(current_dir, index_name)
    index_path = os.path.join(index_dir, "index")
    
    print(f"Checking for index at: {index_path}/index.faiss")
    if os.path.exists(f"{index_path}/index.faiss"):
        print(f"Loading existing vector store from {index_path}/index.faiss")
        try:
            vectorstore = FAISS.load_local(index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
            print("Vector store loaded successfully.")
            return vectorstore
        except Exception as e:
            print(f"Error loading vector store: {e}")
            print("Will create a new vector store.")
    
    print("Creating new vector store...")
    docs = load_documents(sources)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs)
    
    for split in doc_splits:
        if 'source' not in split.metadata:
            split.metadata['source'] = split.metadata.get('source', 'Unknown source')
    
    vectorstore = FAISS.from_documents(doc_splits, OpenAIEmbeddings())
    
    print(f"Saving vector store to {index_path}...")
    try:
        os.makedirs(index_dir, exist_ok=True)
        vectorstore.save_local(index_path)
        print("Vector store saved successfully.")
    except Exception as e:
        print(f"Error saving vector store: {e}")
    
    return vectorstore

# List of URLs and PDF files to load documents from
sources = [
    "https://www.imo.org/en/OurWork/Safety/Pages/DangerousGoods-default.aspx",
    "http://www.imdg.co.kr/sub/page.asp?id=c11&sm=3",
    "./index_imdg/IMDG.pdf"
]

# Create or load vector store
vectorstore = create_or_load_vectorstore(sources)

# Create retriever
retriever = vectorstore.as_retriever(k=10)

# Initialize web search tool
web_search_tool = TavilySearchResults(max_results=5)

# RAG prompt template
rag_prompt = PromptTemplate(
    template="""
    You are an IMDG Code specialist with a deep understanding of the International Maritime Dangerous Goods (IMDG) Code. Your task is to provide detailed and accurate information in response to user questions about IMDG.

    Guidelines for answering:
    1. Use the provided context (Documents) to answer the user's question in detail.
    2. Create a final answer with references, using "SOURCE[number]" in capital letters (e.g., SOURCE[1], SOURCE[2]).
    3. Present information in a clear, concise, and easily understandable manner, using bullet points for organization.
    4. If the question is unclear, politely ask the user for clarification.
    5. For questions about specific UN Numbers:
       - Provide details on: UN No., Proper Shipping Name, Class, Subsidiary hazard, Packing Group, Special Provisions, Limited Quantity, Excepted Quantity, Packing Instructions, Stowage and handling, Segregation, and Properties and observations.
       - If this information is not in the provided context, state that you need to refer to the official IMDG Code for accurate details.
    6. Respond in the language of the user's question. If unable to determine the language, default to English.
    7. If you don't know the answer or if the information is not in the provided context, clearly state "I don't have enough information to answer this question accurately. Please refer to the official IMDG Code or consult with an IMDG expert for the most up-to-date and accurate information."
    8. Limit your response to a maximum of 300 words unless the question specifically requires a longer answer.

    Context format:
    The 'Documents' field contains relevant excerpts from the IMDG Code and related sources. Each document is separated by triple dashes (---) and includes the source information.

    Question: {question}
    Documents: {formatted_documents}
    Answer:
    """,
    input_variables=["question", "formatted_documents"],
)

rag_chain = rag_prompt | llm | StrOutputParser()

# Data model for the output
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        default="no",
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# LLM with tool call
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Grading prompt
system = """You are a teacher grading a quiz. You will be given: 
1/ a QUESTION 
2/ a set of comma separated FACTS provided by the student

You are grading RELEVANCE RECALL:
A score of 1 means that ANY of the FACTS are relevant to the QUESTION. 
A score of 0 means that NONE of the FACTS are relevant to the QUESTION. 
1 is the highest (best) score. 0 is the lowest score you can give. 

Explain your reasoning in a step-by-step manner. Ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "FACTS: \n\n {documents} \n\n QUESTION: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    search: str
    documents: List[str]
    steps: List[str]

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    question = state["question"]
    documents = retriever.invoke(question)
    steps = state["steps"]
    steps.append("retrieve_documents")
    return {"documents": documents, "question": question, "steps": steps}

def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """

    question = state["question"]
    documents = state["documents"]
    
    formatted_documents = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get('source', 'Unknown source')
        formatted_doc = f"Document {i}:\n{doc.page_content}\nSource: {source}\n---"
        formatted_documents.append(formatted_doc)
    
    formatted_documents_str = "\n".join(formatted_documents)
    
    generation = rag_chain.invoke({"formatted_documents": formatted_documents_str, "question": question})
    steps = state["steps"]
    steps.append("generate_answer")
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "steps": steps,
    }

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    question = state["question"]
    documents = state["documents"]
    steps = state["steps"]
    steps.append("grade_document_retrieval")
    filtered_docs = []
    search = "No"
    
    for d in documents:
        try:
            score = retrieval_grader.invoke(
                {"question": question, "documents": d.page_content}
            )
            
            if score is None:
                print(f"Warning: retrieval_grader returned None for document: {d.page_content[:100]}...")
                continue
            
            if not hasattr(score, 'binary_score'):
                print(f"Warning: score object does not have 'binary_score' attribute. Score: {score}")
                continue
            
            grade = score.binary_score
            if grade == "yes":
                filtered_docs.append(d)
            else:
                search = "Yes"
        except Exception as e:
            print(f"Error grading document: {e}")
            print(f"Document content: {d.page_content[:100]}...")
            continue

    return {
        "documents": filtered_docs,
        "question": question,
        "search": search,
        "steps": steps,
    }

def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    question = state["question"]
    documents = state.get("documents", [])
    steps = state["steps"]
    steps.append("web_search")

    try:
        web_results = web_search_tool.invoke({"query": question})
        
        print(f"Web search results type: {type(web_results)}")
        print(f"Web search results: {web_results[:500]}...")  # Print first 500 characters
        
        if isinstance(web_results, str):
            print("Web results is a string. Attempting to parse as JSON...")
            import json
            try:
                web_results = json.loads(web_results)
            except json.JSONDecodeError as e:
                print(f"Failed to parse web search results as JSON: {e}")
                web_results = []
        
        if isinstance(web_results, dict):
            print("Web results is a dictionary. Attempting to extract relevant information...")
            if "results" in web_results:
                web_results = web_results["results"]
            elif "content" in web_results and "url" in web_results:
                web_results = [web_results]
            else:
                print(f"Unexpected dictionary format. Keys: {web_results.keys()}")
                web_results = []
        
        if isinstance(web_results, list):
            for result in web_results:
                if isinstance(result, dict):
                    content = result.get("content") or result.get("snippet")
                    url = result.get("url") or result.get("link")
                    if content and url:
                        documents.append(Document(page_content=content, metadata={"source": url}))
                    else:
                        print(f"Missing content or URL in result: {result}")
                else:
                    print(f"Unexpected result type in list: {type(result)}")
        else:
            print(f"Unexpected web_results type after processing: {type(web_results)}")
        
        print(f"Number of documents after web search: {len(documents)}")
        
    except Exception as e:
        print(f"Error during web search: {e}")
        import traceback
        traceback.print_exc()
    
    return {"documents": documents, "question": question, "steps": steps}

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    search = state["search"]
    if search == "Yes":
        return "search"
    else:
        return "generate"

# Graph
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("web_search", web_search)

# Build graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "search": "web_search",
        "generate": "generate",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

class IMDGSelfLearningAgent:
    def __init__(self):
        self.llm = llm
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self._update_tfidf_matrix()

    def update_knowledge_graph(self, subject, predicate, object):
        with driver.session() as session:
            session.run("""
            MERGE (s:Concept {name: $subject})
            MERGE (o:Concept {name: $object})
            MERGE (s)-[r:RELATION {type: $predicate}]->(o)
            """, subject=subject, predicate=predicate, object=object)
        self._update_tfidf_matrix()

    def _update_tfidf_matrix(self):
        with driver.session() as session:
            result = session.run("""
            MATCH (n)-[r]->(m)
            RETURN n.name as subject, type(r) as predicate, m.name as object
            """)
            documents = [f"{record['subject']} {record['predicate']} {record['object']}" for record in result]
        
        if documents:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            self.tfidf_feature_names = self.tfidf_vectorizer.get_feature_names_out()

    def extract_knowledge(self, response):
        if "New knowledge:" in response:
            knowledge = response.split("New knowledge:")[1].strip()
            try:
                subject, predicate, object = knowledge.split(',')
                self.update_knowledge_graph(subject.strip(), predicate.strip(), object.strip())
                return f"Learned: {subject} {predicate} {object}"
            except:
                return "Failed to extract knowledge"
        return "No new knowledge detected"

    def get_related_knowledge(self, question, limit=5):
        if self.tfidf_matrix is None or self.tfidf_matrix.shape[0] == 0:
            return []
        
        query_vec = self.tfidf_vectorizer.transform([question])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        related_indices = similarities.argsort()[-limit:][::-1]
        
        with driver.session() as session:
            result = session.run("""
            MATCH (n)-[r]->(m)
            RETURN n.name as subject, type(r) as predicate, m.name as object
            """)
            all_triples = [(record['subject'], record['predicate'], record['object']) for record in result]
        
        related_knowledge = [f"{all_triples[i][0]} {all_triples[i][1]} {all_triples[i][2]}" for i in related_indices]
        return related_knowledge

    def query(self, question):
        related_knowledge = self.get_related_knowledge(question)
        context = "\n".join(related_knowledge)

        chat_history = self.memory.chat_memory.messages

        dynamic_prompt = self._generate_dynamic_prompt(question, context, chat_history)

        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        response = app.invoke(
            {
                "question": dynamic_prompt,
                "generation": "",
                "search": "",
                "documents": [],
                "steps": []
            },
            config
        )
        
        ai_response = response.get("generation", "No response generated")
        ai_steps = response.get("steps", [])
        
        learning_result = self.extract_knowledge(ai_response)
        
        self.memory.chat_memory.add_user_message(question)
        self.memory.chat_memory.add_ai_message(ai_response)

        if len(self.memory.chat_memory.messages) % 10 == 0:
            self._integrate_knowledge()

        return ai_response, learning_result, ai_steps

    def _generate_dynamic_prompt(self, question, context, chat_history):
        recent_history = ' '.join([f"{m.type}: {m.content}" for m in chat_history[-5:]])
        return f"""
        Context from knowledge graph:
        {context}

        Recent chat history:
        {recent_history}

        Based on the above context, chat history, and your knowledge of IMDG, please answer the following question:
        {question}

        If you learn any new factual information about IMDG that's not in the context, 
        please include it at the end of your response in the format 'New knowledge: subject, predicate, object'

        Also, if you identify any relationships or patterns in the IMDG data that might be relevant for future questions,
        please mention them in the format 'Insight: your insight here'
        """

    def _integrate_knowledge(self):
        chat_history = self.memory.chat_memory.messages
        full_history = ' '.join([m.content for m in chat_history])
        
        integration_prompt = f"""
        Please analyze the following conversation history about IMDG and extract any additional insights or relationships:
        {full_history}

        Format each insight as 'Insight: subject, predicate, object'
        """
        
        insights = self.llm(integration_prompt)
        for insight in insights.split('\n'):
            if insight.startswith('Insight:'):
                self.extract_knowledge(insight)

    def get_knowledge_summary(self):
        with driver.session() as session:
            result = session.run("""
            MATCH (n:Concept)
            OPTIONAL MATCH (n)-[r]->(m)
            RETURN n.name as concept, collect(distinct type(r) + ' -> ' + m.name) as relations
            LIMIT 10
            """)
            summary = [f"{row['concept']}: {', '.join(row['relations'])}" for row in result]
        return "\n".join(summary)

def visualize_graph():
    with driver.session() as session:
        result = session.run("MATCH (n)-[r]->(m) RETURN n.name, type(r), m.name")
        G = nx.DiGraph()
        for record in result:
            G.add_edge(record["n.name"], record["m.name"], type=record["type(r)"])
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=800, font_size=10, arrows=True)
    edge_labels = nx.get_edge_attributes(G, 'type')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("IMDG Knowledge Graph Visualization")
    return plt

def main():
    st.title("IMDG Self-Learning and Knowledge Graph Building Agent")

    agent = IMDGSelfLearningAgent()

    st.subheader("Current IMDG Knowledge Summary")
    st.write(agent.get_knowledge_summary())

    question = st.text_input("Ask a question about IMDG:")
    if st.button("Submit Question"):
        response, learning_result, steps = agent.query(question)
        st.write("Answer:", response)
        st.write(learning_result)

        st.subheader("Steps Taken")
        for step in steps:
            st.write(f"- {step}")

        st.subheader("Related Knowledge")
        related_knowledge = agent.get_related_knowledge(question)
        if related_knowledge:
            for knowledge in related_knowledge:
                st.write(f"- {knowledge}")
        
        st.subheader("Updated IMDG Knowledge Summary")
        st.write(agent.get_knowledge_summary())

    if st.button("Visualize IMDG Knowledge Graph"):
        plt = visualize_graph()
        st.pyplot(plt)

    if st.button("IMDG Knowledge Graph Statistics"):
        with driver.session() as session:
            result = session.run("""
            MATCH (n)
            OPTIONAL MATCH ()-[r]->()
            RETURN 
                count(distinct n) as node_count,
                count(distinct r) as relation_count,
                count(distinct labels(n)) as label_count
            """)
            stats = result.single()
            st.write(f"Number of nodes: {stats['node_count']}")
            st.write(f"Number of relationships: {stats['relation_count']}")
            st.write(f"Number of unique labels: {stats['label_count']}")

if __name__ == "__main__":
    main()