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
import networkx as nx
import matplotlib.pyplot as plt

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
            allow_dangerous_code=True,
            handle_parsing_errors=True)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def update_knowledge_graph(self, subject, predicate, object):
        with driver.session() as session:
            session.run("""
            MERGE (s:Concept {name: $subject})
            MERGE (o:Concept {name: $object})
            MERGE (s)-[r:RELATION {type: $predicate}]->(o)
            """, subject=subject, predicate=predicate, object=object)

    def extract_knowledge(self, response):
        if "New knowledge:" in response:
            knowledge = response.split("New knowledge:")[1].strip()
            try:
                subject, predicate, object = knowledge.split(',')
                self.update_knowledge_graph(subject.strip(), predicate.strip(), object.strip())
                return f"학습됨: {subject} {predicate} {object}"
            except:
                return "지식 추출 실패"
        return "새로운 지식이 감지되지 않았습니다"

    def get_related_knowledge(self, question, limit=5):
        with driver.session() as session:
            result = session.run("""
            MATCH (n:Concept)
            WHERE n.name CONTAINS $keyword
            MATCH (n)-[r]-(m)
            RETURN n.name, type(r), m.name
            LIMIT $limit
            """, keyword=question.lower(), limit=limit)
            return [f"{row['n.name']} {row['type(r)']} {row['m.name']}" for row in result]

    def query(self, question):
        related_knowledge = self.get_related_knowledge(question)
        context = "\n".join(related_knowledge)

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

def visualize_graph():
    with driver.session() as session:
        result = session.run("MATCH (n)-[r]->(m) RETURN n.name, type(r), m.name")
        G = nx.DiGraph()
        for record in result:
            G.add_edge(record["n.name"], record["m.name"], type=record["type(r)"])
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=8, arrows=True)
    edge_labels = nx.get_edge_attributes(G, 'type')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Knowledge Graph Visualization")
    return plt

def main():
    st.title("CSV 기반 자기 학습 및 지식 그래프 구축 에이전트")

    uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type="csv")
    
    if uploaded_file is not None:
        agent = CSVKnowledgeAgent(uploaded_file)
        st.success("CSV 파일이 성공적으로 로드되었습니다.")

        st.subheader("CSV 파일 정보")
        st.write(agent.df.head())
        st.write(f"총 행 수: {len(agent.df)}, 컬럼: {', '.join(agent.df.columns)}")

        question = st.text_input("CSV 데이터에 대해 질문하세요:")
        if st.button("질문하기"):
            response, learning_result = agent.query(question)
            st.write("답변:", response)
            st.write(learning_result)

            related_knowledge = agent.get_related_knowledge(question)
            if related_knowledge:
                st.subheader("관련 지식")
                for knowledge in related_knowledge:
                    st.write(f"- {knowledge}")

        if st.button("지식 그래프 시각화"):
            plt = visualize_graph()
            st.pyplot(plt)

        if st.button("지식 그래프 통계"):
            with driver.session() as session:
                result = session.run("""
                MATCH (n)
                OPTIONAL MATCH ()-[r]->()
                RETURN 
                    count(distinct n) as node_count,
                    count(distinct r) as relation_count,
                    size(apoc.coll.toSet(labels(n))) as label_count
                """)
                stats = result.single()
                st.write(f"노드 수: {stats['node_count']}")
                st.write(f"관계 수: {stats['relation_count']}")
                st.write(f"레이블 종류 수: {stats['label_count']}")

if __name__ == "__main__":
    main()