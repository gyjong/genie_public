from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import re
import requests

llm = ChatOpenAI(model="gpt-4o-mini")

def validate_links(text):
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    valid_links = []
    invalid_links = []
    for url in urls:
        try:
            response = requests.head(url, timeout=5)
            if response.status_code == 200:
                valid_links.append(url)
            else:
                invalid_links.append(url)
        except:
            invalid_links.append(url)
    return valid_links, invalid_links, urls

def validate_content(report, language):
    valid_links, invalid_links, all_links = validate_links(report)
    
    if language == 'ko':
        system_message = SystemMessage(content="당신은 한국어로 작성된 보고서를 검증하고 개선하는 전문가입니다. 주어진 보고서의 정확성과 완성도를 평가하고, 필요한 경우 수정하세요.")
        human_message = HumanMessage(content=f"""다음 보고서의 정확성과 완성도를 검증하고 개선해주세요:

        {report}

        다음 측면을 고려하여 보고서를 평가하고 개선해주세요:
        1. 쿼리와의 관련성
        2. 정보의 정확성
        3. 보고서의 완성도
        4. 명확성과 구조
        5. 링크의 유효성 (유효한 링크: {valid_links}, 유효하지 않은 링크: {invalid_links})

        유효하지 않은 링크는 제거하거나 대체해주세요.

        다음 형식으로 응답해주세요:
        개선된 보고서:
        [전체 개선된 보고서를 마크다운 형식으로 작성]

        평가:
        [위에서 언급한 측면들을 기반으로 한 간단한 보고서 평가 및 개선 사항]

        링크 목록:
        [모든 링크를 나열하고 각 링크의 유효성 여부를 표시]
        """)
    else:
        system_message = SystemMessage(content="You are an expert in validating and improving reports written in English. Evaluate the accuracy and completeness of the given report, and make necessary corrections and improvements.")
        human_message = HumanMessage(content=f"""Please validate and improve the following report for accuracy and completeness:

        {report}

        Evaluate and improve the report considering the following aspects:
        1. Relevance to the query
        2. Accuracy of information
        3. Completeness of the report
        4. Clarity and structure
        5. Validity of links (Valid links: {valid_links}, Invalid links: {invalid_links})

        Remove or replace any invalid links.

        Format your response as follows:
        Improved Report:
        [The entire improved report in Markdown format]

        Evaluation:
        [A brief evaluation of the report based on the aspects mentioned above, including improvements made]

        Link List:
        [List all links and indicate their validity]
        """)

    response = llm.invoke([system_message, human_message])
    content = response.content

    # Split the content into report, evaluation, and link list
    if language == 'ko':
        parts = content.split("평가:")
        improved_report = parts[0].replace("개선된 보고서:", "").strip()
        evaluation_and_links = parts[1].split("링크 목록:")
    else:
        parts = content.split("Evaluation:")
        improved_report = parts[0].replace("Improved Report:", "").strip()
        evaluation_and_links = parts[1].split("Link List:")

    evaluation = evaluation_and_links[0].strip()
    link_list = evaluation_and_links[1].strip() if len(evaluation_and_links) > 1 else ""

    return {"validated_report": improved_report, "evaluation": evaluation, "link_list": link_list, "all_links": all_links}