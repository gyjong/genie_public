import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.schema import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.graphs import Neo4jGraph
from langchain.agents.agent_types import AgentType
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

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

class CSVKnowledgeAgent:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.agent = create_pandas_dataframe_agent(
            self.llm, 
            self.df, 
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            allow_dangerous_code=True)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def update_knowledge_graph(self, subject, predicate, object):
        with driver.session() as session:
            session.run("""
            MERGE (s:Concept {name: $subject})
            MERGE (o:Concept {name: $object})
            MERGE (s)-[r:RELATION {type: $predicate}]->(o)
            """, subject=subject, predicate=predicate, object=object)

    def extract_knowledge(self, response):
        # This is a simplified extraction. In practice, you might use NLP or another LLM call.
        if "New knowledge:" in response:
            knowledge = response.split("New knowledge:")[1].strip()
            try:
                subject, predicate, object = knowledge.split(',')
                self.update_knowledge_graph(subject.strip(), predicate.strip(), object.strip())
                return f"학습됨: {subject} {predicate} {object}"
            except:
                return "지식 추출 실패"
        return "새로운 지식이 감지되지 않았습니다"

    def query(self, question):
        # Retrieve relevant knowledge from the graph
        with driver.session() as session:
            result = session.run("MATCH (n)-[r]->(m) RETURN n.name, type(r), m.name LIMIT 5")
            context = "\n".join([f"{row['n.name']} {row['type(r)']} {row['m.name']}" for row in result])

        # Combine the question with the context and instructions
        full_query = f"""
        Context from knowledge graph:
        {context}

        Based on the above context and the CSV data, please answer the following question:
        {question}

        If you learn any new factual information from the CSV that's not in the context, 
        please include it at the end of your response in the format 'New knowledge: subject, predicate, object'
        """

        response = self.agent.run(full_query)
        learning_result = self.extract_knowledge(response)
        return response, learning_result

def main():
    st.title("CSV 기반 자기 학습 및 지식 그래프 구축 에이전트")

    uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type="csv")
    
    if uploaded_file is not None:
        agent = CSVKnowledgeAgent(uploaded_file)
        st.success("CSV 파일이 성공적으로 로드되었습니다.")

        question = st.text_input("CSV 데이터에 대해 질문하세요:")
        if st.button("질문하기"):
            response, learning_result = agent.query(question)
            st.write("답변:", response)
            st.write(learning_result)

        if st.button("현재 지식 그래프 보기"):
            with driver.session() as session:
                result = session.run("MATCH (n)-[r]->(m) RETURN n.name AS source, type(r) AS relation, m.name AS target")
                graph_data = [{"source": row["source"], "relation": row["relation"], "target": row["target"]} for row in result]
            
            st.json(graph_data)

if __name__ == "__main__":
    main()