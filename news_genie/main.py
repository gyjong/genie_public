import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agents.web_search import web_search
from agents.report_generator import generate_report
from agents.content_validator import validate_content
from utils.vector_store import save_to_vector_store, search_vector_store
import base64
import re

load_dotenv()

PORTS = ["NINGBO", "SHANGHAI", "YANTIAN", "SINGAPORE", "TANGER", "LE HAVRE", "HAMBURG", "GDANSK", "ROTTERDAM", "ALGECIRAS", "PORT KLANG"]
VESSELS = [
    "APL CHANGI", "APL MERLION", "APL RAFFLES", "APL SINGAPURA", "APL TEMASEK",
    "APL VANDA", "CMA CGM ALEXANDER VON HUMBOLDT", "CMA CGM BENJAMIN FRANKLIN",
    "CMA CGM BOUGAINVILLE", "CMA CGM EOURES", "CMA CGM GEORG FORSTER",
    "CMA CGM GRACE BAY", "CMA CGM ROQUEVAIRE", "CMA CGM VASCO DE GAMA",
    "CMA CGM ZHENG HE"
]

llm = ChatOpenAI(model="gpt-4o-mini")

def is_korean(text):
    return any('\uac00' <= char <= '\ud7a3' for char in text)

def process_query(user_input, search_option, language):
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_status(message):
        status_text.text(message)

    try:
        # Vector DB retrieval
        status_text.text("벡터 데이터베이스에서 관련 정보를 검색 중입니다..." if language == 'ko' else "Retrieving relevant information from vector database...")
        vector_results = search_vector_store(user_input)
        progress_bar.progress(25)
        st.text("벡터 DB 검색이 완료되었습니다." if language == 'ko' else "Vector DB retrieval completed.")
        
        if vector_results:
            st.text(f"{'벡터 DB에서 검색된 결과:' if language == 'ko' else 'Results from Vector DB:'} {len(vector_results)} items found")
            for idx, result in enumerate(vector_results[:3]):
                st.text(f"{'결과' if language == 'ko' else 'Result'} {idx + 1}: {result[:300]}...")

        if search_option == "Web Search":
            # Web search
            status_text.text("웹 검색을 시작합니다..." if language == 'ko' else "Initiating web search...")
            search_results = web_search(user_input, PORTS, VESSELS, update_status)
            st.text(f"{'웹 검색 결과:' if language == 'ko' else 'Web search results:'} {len(search_results['search_results'])} items found")
            if isinstance(search_results, dict) and 'search_results' in search_results:
                for idx, result in enumerate(search_results['search_results'][:5]):
                    content = result.get('content', 'No content')
                    if 'error' in content.lower() or 'failed' in content.lower():
                        st.warning(f"{'결과' if language == 'ko' else 'Result'} {idx + 1}: {content}")
                    else:
                        st.text(f"{'결과' if language == 'ko' else 'Result'} {idx + 1}: {content[:100]}...")
            progress_bar.progress(50)
            st.text("웹 검색이 완료되었습니다." if language == 'ko' else "Web search completed.")

            # Combine results
            combined_results = search_results["search_results"]
            if vector_results:
                combined_results.extend([{"content": result} for result in vector_results])
        else:
            # Use only vector DB results
            combined_results = [{"content": result} for result in vector_results]

        # Generate report
        status_text.text("보고서를 생성 중입니다..." if language == 'ko' else "Generating report...")
        report = generate_report(combined_results, language)
        progress_bar.progress(75)
        st.text("보고서 생성이 완료되었습니다." if language == 'ko' else "Report generation completed.")

        # Validate content
        status_text.text("내용을 검증 중입니다..." if language == 'ko' else "Validating content...")
        validation_result = validate_content(report, language)
        validated_report = validation_result["validated_report"]
        evaluation = validation_result["evaluation"]
        all_links = validation_result["all_links"]
        progress_bar.progress(100)
        status_text.text("처리가 완료되었습니다." if language == 'ko' else "Process completed.")

        return validated_report, evaluation, language, all_links

    except Exception as e:
        st.error(f"처리 중 오류가 발생했습니다: {str(e)}" if language == 'ko' else f"An error occurred during processing: {str(e)}")
        import traceback
        st.error(f"추적: {traceback.format_exc()}" if language == 'ko' else f"Traceback: {traceback.format_exc()}")
        return None, None, language, None

def main():
    # 폰트 파일을 base64로 인코딩
    with open("./fonts/Freesentation.ttf", "rb") as font_file:
        font_base64 = base64.b64encode(font_file.read()).decode()

    # CSS를 사용하여 폰트 적용
    st.markdown(
        f"""
        <style>
        @font-face {{
            font-family: 'Freesentation';
            src: url(data:font/ttf;base64,{font_base64}) format('truetype');
        }}
        html, body, [class*="st-"] {{
            font-family: 'Freesentation', sans-serif;
        }}
        .report-content {{
            background-color: #f0f0f0;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .stMarkdown {{
            font-size: 16px;
            line-height: 1.6;
        }}
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        .stMarkdown ul, .stMarkdown ol {{
            margin-left: 20px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Maritime News Genie")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "language" not in st.session_state:
        st.session_state.language = 'en'

    if "search_option" not in st.session_state:
        st.session_state.search_option = "Vector DB Only"

    # 검색 옵션 선택 (항상 표시)
    search_option = st.radio(
        "Select search option:",
        ("Vector DB Only", "Web Search"),
        key="search_option"
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt_text = "What would you like to know about ports and vessels?"

    if prompt := st.chat_input(prompt_text):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 언어 감지
        st.session_state.language = 'ko' if is_korean(prompt) else 'en'

        try:
            # 이전 대화 내용을 포함하여 쿼리 생성
            full_context = "\n".join([m["content"] for m in st.session_state.messages])
            
            report, evaluation, _, all_links = process_query(full_context, search_option, st.session_state.language)

            if report and evaluation:
                with st.chat_message("assistant"):
                    st.markdown("### Report" if st.session_state.language == 'en' else "### 보고서")
                    st.markdown(report, unsafe_allow_html=True)
                    st.markdown("### Evaluation" if st.session_state.language == 'en' else "### 평가")
                    st.markdown(evaluation, unsafe_allow_html=True)
                    
                    # SOURCES 정보 표시
                    st.markdown("### Sources" if st.session_state.language == 'en' else "### 출처")
                    sources = re.findall(r'SOURCES\[(\d+)\]\s*\((.*?)\)', report)
                    for idx, (source_num, url) in enumerate(sources, 1):
                        st.markdown(f"{idx}. SOURCES[{source_num}]: [{url}]({url})")
                    
                    # 링크 목록 표시
                    st.markdown("### Link List" if st.session_state.language == 'en' else "### 링크 목록")
                    for link in all_links:
                        st.markdown(f"- {link}")
                
                st.session_state.messages.append({"role": "assistant", "content": f"{'Report' if st.session_state.language == 'en' else '보고서'}:\n{report}\n\n{'Evaluation' if st.session_state.language == 'en' else '평가'}:\n{evaluation}"})

                # Save the report to vector store
                save_to_vector_store(report)
                st.success("The report has been saved to the vector database." if st.session_state.language == 'en' else "보고서가 벡터 데이터베이스에 저장되었습니다.")
            else:
                st.error("Failed to generate report and evaluation." if st.session_state.language == 'en' else "보고서와 평가를 생성하지 못했습니다.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}" if st.session_state.language == 'en' else f"오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    main()