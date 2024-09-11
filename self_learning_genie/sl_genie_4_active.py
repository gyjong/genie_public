import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.schema import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain_community.graphs import Neo4jGraph
from langchain.agents.agent_types import AgentType
from neo4j import GraphDatabase
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load environment variables
load_dotenv()

# LangSmith 
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "self-learning-agent"

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

class CSVKnowledgeAgent:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.llm = llm
        self.agent = create_pandas_dataframe_agent(
            self.llm, 
            self.df, 
            verbose=True,
            # agent_type=AgentType.OPENAI_FUNCTIONS,
            allow_dangerous_code=True)
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

        response = self.agent.run(dynamic_prompt)
        learning_result = self.extract_knowledge(response)
        
        self.memory.chat_memory.add_user_message(question)
        self.memory.chat_memory.add_ai_message(response)

        if len(self.memory.chat_memory.messages) % 10 == 0:
            self._integrate_knowledge()

        return response, learning_result

    def _generate_dynamic_prompt(self, question, context, chat_history):
        recent_history = ' '.join([f"{m.type}: {m.content}" for m in chat_history[-5:]])
        return f"""
        Context from knowledge graph:
        {context}

        Recent chat history:
        {recent_history}

        Based on the above context, chat history, and the CSV data, please answer the following question:
        {question}

        If you learn any new factual information from the CSV that's not in the context, 
        please include it at the end of your response in the format 'New knowledge: subject, predicate, object'

        Also, if you identify any relationships or patterns in the data that might be relevant for future questions,
        please mention them in the format 'Insight: your insight here'
        """

    def _integrate_knowledge(self):
        chat_history = self.memory.chat_memory.messages
        full_history = ' '.join([m.content for m in chat_history])
        
        integration_prompt = f"""
        Please analyze the following conversation history and extract any additional insights or relationships:
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
    
        # Try to set a font that supports Korean characters
        try:
            font_path = '/System/Library/Fonts/AppleSDGothicNeo.ttc'  # Path to a Korean font on macOS
            prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = 'AppleGothic'  # Set font family
        except:
            print("Korean font not found. Using default font.")

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=800, font_size=10, arrows=True)
    edge_labels = nx.get_edge_attributes(G, 'type')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Knowledge Graph Visualization")
    return plt

def main():
    st.title("CSV-based Self-Learning and Knowledge Graph Building Agent")

    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    
    if uploaded_file is not None:
        agent = CSVKnowledgeAgent(uploaded_file)
        st.success("CSV file successfully loaded.")

        st.subheader("CSV File Information")
        st.write(agent.df.head())
        st.write(f"Total rows: {len(agent.df)}, Columns: {', '.join(agent.df.columns)}")

        st.subheader("Current Knowledge Summary")
        st.write(agent.get_knowledge_summary())

        question = st.text_input("Ask a question about the CSV data:")
        if st.button("Submit Question"):
            response, learning_result = agent.query(question)
            st.write("Answer:", response)
            st.write(learning_result)

            st.subheader("Related Knowledge")
            related_knowledge = agent.get_related_knowledge(question)
            if related_knowledge:
                for knowledge in related_knowledge:
                    st.write(f"- {knowledge}")
            
            st.subheader("Updated Knowledge Summary")
            st.write(agent.get_knowledge_summary())

            st.subheader("Knowledge Graph Visualization")
            plt = visualize_graph()
            st.pyplot(plt)

        if st.button("Visualize Knowledge Graph"):
            plt = visualize_graph()
            st.pyplot(plt)

        if st.button("Knowledge Graph Statistics"):
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