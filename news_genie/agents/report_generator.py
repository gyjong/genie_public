from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import json

llm = ChatOpenAI(model="gpt-4o-mini")

def generate_report(search_results, language):
    system_prompt = """You are an expert in creating comprehensive reports about ports and vessels based on given information. 
    Create a final answer with references ("SOURCES[number]"). Use "SOURCES[number]" in capital letters regardless of the number of sources you use.
    For each SOURCES reference, include the URL for web pages in parentheses. For example: SOURCES[2] (https://example.com)."""

    if language == 'ko':
        system_message = SystemMessage(content=system_prompt + " 보고서를 한국어로 작성하세요.")
        human_message = HumanMessage(content=f"""다음 검색 결과를 바탕으로 항구와 선박에 대한 종합적인 보고서를 작성해주세요:

        {json.dumps(search_results, indent=2, ensure_ascii=False)}

        언급된 각 항구와 선박에 대한 관련 정보를 포함해주세요. 보고서를 마크다운 형식으로 작성하고 SOURCES 참조를 포함해주세요.
        JSON 형식이 아닌 순수한 마크다운 텍스트로 작성해 주세요.
        """)
    else:
        system_message = SystemMessage(content=system_prompt)
        human_message = HumanMessage(content=f"""Based on the following search results, generate a comprehensive report about ports and vessels:

        {json.dumps(search_results, indent=2)}

        Include relevant information for each port and vessel mentioned. Format the report in markdown and include SOURCES references.
        Please write in pure markdown text, not in JSON format.
        """)

    response = llm.invoke([system_message, human_message])
    report = response.content

    return report  # 직접 마크다운 텍스트 반환