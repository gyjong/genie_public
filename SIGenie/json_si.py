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
COLLECTION_NAME = "si"

# MongoDB 클라이언트 생성
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

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
    """중첩된 딕셔너리와 리스트를 위한 재귀적 입력 필드 생성 함수"""
    updated_data = {}
    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{prefix}{key}"
            if isinstance(value, dict):
                updated_data[key] = create_input_fields(value, f"{full_key}.")
            elif isinstance(value, list):
                updated_data[key] = create_list_input_fields(value, full_key)
            else:
                updated_data[key] = st.text_input(full_key, str(value))
    elif isinstance(data, list):
        return create_list_input_fields(data, prefix)
    else:
        return st.text_input(prefix, str(data))
    return updated_data

def create_list_input_fields(data_list, prefix):
    """리스트 형태의 데이터를 위한 입력 필드 생성 함수"""
    updated_list = []
    for i, item in enumerate(data_list):
        if isinstance(item, dict):
            with st.expander(f"{prefix} Item {i+1}"):
                updated_item = create_input_fields(item, f"{prefix}.{i}.")
        else:
            updated_item = st.text_input(f"{prefix}.{i}", str(item))
        updated_list.append(updated_item)
    return updated_list

def main():
    st.title("Shipping Instruction JSON Editor")

    # JSON 파일 로드
    json_files = load_json_files('./si/')

    # MongoDB에 저장 (이미 저장되어 있지 않은 경우에만)
    for filename, data in json_files.items():
        existing = collection.find_one({'bookingReference': data['bookingReference']})
        if not existing:
            save_to_mongodb(data)

    # MongoDB에서 모든 문서 가져오기
    documents = list(collection.find())

    # 문서 선택
    selected_doc = st.selectbox(
        "Edit Shipping Instruction",
        options=documents,
        format_func=lambda x: x['bookingReference']
    )

    if selected_doc:
        st.write("---")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.subheader("Voyage & Route Details")
            voyage_details = create_input_fields(selected_doc.get('voyageDetails', {}), 'voyageDetails.')
            route_details = create_input_fields(selected_doc.get('routeDetails', {}), 'routeDetails.')
            
            st.subheader("Payment & Documentation")
            payment_details = create_input_fields(selected_doc.get('paymentDetails', {}), 'paymentDetails.')
            doc_details = create_input_fields(selected_doc.get('documentationDetails', {}), 'documentationDetails.')

        with col2:
            st.subheader("Party Details")
            party_details = create_input_fields(selected_doc.get('partyDetails', {}), 'partyDetails.')

        with col3:
            st.subheader("Shipping Information")
            shipping_term = st.text_input("shippingTerm", selected_doc.get('shippingTerm', ''))
            hs_code = st.text_input("hsCode", selected_doc.get('hsCode', ''))
            commodity_description = st.text_area("commodityDescription", selected_doc.get('commodityDescription', ''))
            
            st.subheader("Containers")
            containers = create_input_fields(selected_doc.get('containers', []), 'containers')
            
            st.subheader("Total Shipment")
            total_shipment = create_input_fields(selected_doc.get('totalShipment', {}), 'totalShipment.')

        with col4:
            st.subheader("Additional Information")
            additional_info = create_input_fields(selected_doc.get('additionalInformation', {}), 'additionalInformation.')

        # Special Cargo 정보 추가 및 수정 가능하게 변경
        st.write("---")
        st.subheader("Special Cargo Information")

        # Out of Gauge Dimensions
        oog = selected_doc.get('outOfGaugeDimensions')
        if oog:
            st.write("Out of Gauge Dimensions:")
            oog_updated = {}
            for key in ['length', 'width', 'height', 'overWidth', 'overHeight']:
                value = oog.get(key, '')
                if value == 'In-Gauge':
                    oog_updated[key] = st.text_input(f"{key.capitalize()} (mm)", value=value)
                else:
                    try:
                        numeric_value = float(value) if value else 0
                        oog_updated[key] = st.number_input(f"{key.capitalize()} (mm)", value=numeric_value)
                    except ValueError:
                        oog_updated[key] = st.text_input(f"{key.capitalize()} (mm)", value=value)

        # Dangerous Goods
        dg = selected_doc.get('dangerousGoods')
        if dg:
            st.write("Dangerous Goods:")
            dg_updated = {}
            dg_updated['containerNumber'] = st.text_input("Container Number (DG)", value=dg.get('containerNumber', ''))
            dg_updated['unClass'] = st.text_input("UN Class", value=dg.get('unClass', ''))
            dg_updated['unCode'] = st.text_input("UN Code", value=dg.get('unCode', ''))
            dg_updated['hsCode'] = st.text_input("HS Code (DG)", value=dg.get('hsCode', ''))
            dg_updated['flashPoint'] = st.text_input("Flash Point", value=dg.get('flashPoint', ''))
            dg_updated['additionalInfo'] = st.text_area("Additional Info (DG)", value=dg.get('additionalInfo', ''))

        # Reefer Settings
        rs = selected_doc.get('reeferSettings')
        if rs:
            st.write("Reefer Settings:")
            rs_updated = {}
            rs_updated['containerNumber'] = st.text_input("Container Number (Reefer)", value=rs.get('containerNumber', ''))
            rs_updated['temperature'] = st.text_input("Temperature", value=rs.get('temperature', ''))
            rs_updated['minTemperature'] = st.text_input("Min Temperature", value=rs.get('minTemperature', ''))
            rs_updated['maxTemperature'] = st.text_input("Max Temperature", value=rs.get('maxTemperature', ''))
            rs_updated['ventilation'] = st.text_input("Ventilation", value=rs.get('ventilation', ''))
            rs_updated['humidity'] = st.text_input("Humidity", value=rs.get('humidity', ''))

        if st.button("Update"):
            # 업데이트된 데이터 수집
            updated_data = {
                'voyageDetails': voyage_details,
                'routeDetails': route_details,
                'paymentDetails': payment_details,
                'documentationDetails': doc_details,
                'partyDetails': party_details,
                'shippingTerm': shipping_term,
                'hsCode': hs_code,
                'commodityDescription': commodity_description,
                'containers': containers,
                'totalShipment': total_shipment,
                'additionalInformation': additional_info,
            }
            
            # 특수화물 정보가 있을 경우에만 업데이트 데이터에 추가
            if oog:
                updated_data['outOfGaugeDimensions'] = oog_updated
            if dg:
                updated_data['dangerousGoods'] = dg_updated
            if rs:
                updated_data['reeferSettings'] = rs_updated

            # MongoDB 업데이트
            update_mongodb(selected_doc['_id'], updated_data)
            st.success("Document updated successfully!")

    # Footer 제거

if __name__ == "__main__":
    main()