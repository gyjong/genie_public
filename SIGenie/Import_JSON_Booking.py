import os
import json
import streamlit as st
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# MongoDB 연결 설정
MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB_NAME")
COLLECTION_NAME = "bkg"

# MongoDB 클라이언트 생성
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Streamlit 페이지 설정
st.set_page_config(layout="wide")

# CSS를 사용하여 폰트 적용
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Avenir:wght@400;700&display=swap');
    html, body, [class*="st-"] {
        font-family: 'Avenir', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

def load_json_files(directory):
    """지정된 디렉토리에서 모든 JSON 파일을 로드합니다."""
    json_files = {}
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                json_files[filename] = json.load(file)
    return json_files

def save_to_mongodb(json_data):
    """JSON 데이터를 MongoDB에 저장합니다."""
    result = collection.insert_one(json_data)
    return result.inserted_id

def update_mongodb(doc_id, updated_data):
    """MongoDB의 문서를 업데이트합니다."""
    collection.update_one({'_id': ObjectId(doc_id)}, {'$set': updated_data})

def create_input_fields(data, prefix=''):
    """중첩된 딕셔너리를 위한 재귀적 입력 필드 생성 함수"""
    updated_data = {}
    for key, value in data.items():
        full_key = f"{prefix}{key}"
        if isinstance(value, dict):
            updated_data[key] = create_input_fields(value, f"{full_key}.")
        else:
            updated_data[key] = st.text_input(full_key, str(value))
    return updated_data

def main():
    st.title("Booking JSON Editor")

    # JSON 파일 로드
    json_files = load_json_files('./bkg/')

    # MongoDB에 저장 (이미 저장되어 있지 않은 경우에만)
    for filename, data in json_files.items():
        existing = collection.find_one({'bookingReference': data['bookingReference']})
        if not existing:
            save_to_mongodb(data)

    # MongoDB에서 모든 문서 가져오기
    documents = list(collection.find())

    # 문서 선택
    selected_doc = st.selectbox(
        "Edit Booking",
        options=documents,
        format_func=lambda x: x['bookingReference']
    )

    if selected_doc:
        st.write("---")
        
        # Header Section
        st.subheader("Booking Details")
        header_fields = ['bookingReference', 'customerName', 'shipperName', 'invoiceReceiver', 'shippingTerm', 'remarks']
        header_data = {field: selected_doc.get(field, '') for field in header_fields}
        header_inputs = create_input_fields(header_data)

        # Main Content
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Voyage & Route Details")
            voyage_details = create_input_fields(selected_doc.get('voyageDetails', {}), 'voyageDetails.')
            route_details = create_input_fields(selected_doc.get('routeDetails', {}), 'routeDetails.')
            schedule_details = create_input_fields(selected_doc.get('scheduleDetails', {}), 'scheduleDetails.')

        with col2:
            st.subheader("Cargo & Container Details")
            cargo_details = create_input_fields(selected_doc.get('cargoDetails', {}), 'cargoDetails.')
            container_details = create_input_fields(selected_doc.get('containerDetails', {}), 'containerDetails.')
            empty_pickup = st.text_input('emptyContainerPickupLocation', selected_doc.get('emptyContainerPickupLocation', ''))

        if st.button("Update"):
            # 업데이트된 데이터 수집
            updated_data = {
                **header_inputs,
                'voyageDetails': voyage_details,
                'routeDetails': route_details,
                'scheduleDetails': schedule_details,
                'cargoDetails': cargo_details,
                'containerDetails': container_details,
                'emptyContainerPickupLocation': empty_pickup
            }
            # MongoDB 업데이트
            update_mongodb(selected_doc['_id'], updated_data)
            st.success("Document updated successfully!")

if __name__ == "__main__":
    main()