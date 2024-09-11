from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import os
import requests
import time

load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_search = TavilySearchResults(api_key=tavily_api_key)

def web_search(query, ports, vessels, status_callback):
    all_results = []
    max_query_length = 300  # 쿼리 길이 제한
    max_retries = 3  # 최대 재시도 횟수
    retry_delay = 5  # 재시도 사이의 대기 시간(초)

    try:
        total_items = len(ports) + len(vessels)
        current_item = 0

        # 각 포트에 대해 검색
        for port in ports:
            current_item += 1
            port_query = f"{query} related to port {port}"
            port_query = port_query[:max_query_length]
            status_callback(f"Searching for port: {port} ({current_item}/{total_items})")
            port_results = search_with_retry(port_query, max_retries, retry_delay)
            all_results.extend(process_results(port_results, [port], vessels))

        # 각 선박에 대해 검색
        for vessel in vessels:
            current_item += 1
            vessel_query = f"{query} related to vessel {vessel}"
            vessel_query = vessel_query[:max_query_length]
            status_callback(f"Searching for vessel: {vessel} ({current_item}/{total_items})")
            vessel_results = search_with_retry(vessel_query, max_retries, retry_delay)
            all_results.extend(process_results(vessel_results, ports, [vessel]))

        return {"search_results": all_results}
    except Exception as err:
        error_message = f"An error occurred: {err}"
        status_callback(error_message)
        return {"search_results": [{"content": error_message}]}

def search_with_retry(query, max_retries, retry_delay):
    for attempt in range(max_retries):
        try:
            return tavily_search.invoke(query)
        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 429:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
            raise
    return [{"content": f"Failed to retrieve results after {max_retries} attempts"}]

def process_results(search_results, ports, vessels):
    formatted_results = []
    if isinstance(search_results, list):
        for result in search_results:
            if isinstance(result, dict):
                content = result.get("content", "")
                if any(port.lower() in content.lower() for port in ports) or \
                   any(vessel.lower() in content.lower() for vessel in vessels):
                    formatted_results.append({"content": content})
    elif isinstance(search_results, dict):
        content = search_results.get("content", "")
        if any(port.lower() in content.lower() for port in ports) or \
           any(vessel.lower() in content.lower() for vessel in vessels):
            formatted_results.append({"content": content})
    else:
        formatted_results.append({"content": str(search_results)})
    return formatted_results