import streamlit as st
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import base64

# Load environment variables and set up MongoDB connection
load_dotenv()
MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB_NAME")
COLLECTION_NAME = "bl"
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Set page to wide mode
st.set_page_config(layout="wide")

# Custom CSS to style the BL form
custom_css = """
<style>
    .bl-form {
        font-family: Arial, sans-serif;
        border: 2px solid black;
        padding: 10px;
        margin-bottom: 20px;
        width: 100%;
    }
    .bl-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        border-bottom: 1px solid black;
        padding-bottom: 10px;
        margin-bottom: 10px;
    }
    .bl-section {
        margin-bottom: 10px;
        border: 1px solid black;
        padding: 5px;
    }
    .bl-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
    }
    .bl-footer {
        border-top: 1px solid black;
        padding-top: 10px;
        margin-top: 10px;
    }
    .bl-logo {
        text-align: right;
        margin-left: auto;
    }
    .bl-logo img {
        max-width: 250px;
        height: auto;
    }
    .bl-table {
        width: 100%;
        border-collapse: collapse;
    }
    .bl-table th, .bl-table td {
        border: 1px solid black;
        padding: 5px;
        text-align: left;
    }
</style>
"""

def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def display_bl_form(doc):
    # Apply custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)
    
    # Load and encode the logo
    logo_base64 = get_base64_encoded_image("./img/containergenie.png")
    
    # Adjust the HTML generation with a separate function call outside the HTML string
    container_info_html = generate_container_rows(doc['containers'], doc)

    # Create the BL form HTML
    bl_html = f"""
    <div class="bl-form">
        <div class="bl-header">
            <div>
                <h2>BILL OF LADING</h2>
            </div>
            <div>
                <p><strong>BOOKING NO:</strong> {doc.get('bookingReference', '')}</p>
                <p><strong>SERVICE:</strong> {doc.get('service', '')}</p>
                <p><strong>BL NO:</strong> {doc.get('bookingReference', '')}</p>
            </div>
            <div class="bl-logo">
                <img src="data:image/jpeg;base64,{logo_base64}" alt="Company Logo">
            </div>
        </div>
        <div class="bl-section">
            <h3>SHIPPER (NAME AND FULL ADDRESS)</h3>
            <p>{doc.get('partyDetails', {}).get('shipper', {}).get('name', '')}</p>
            <p>{doc.get('partyDetails', {}).get('shipper', {}).get('address', '')}</p>
            <p>Tel: {doc.get('partyDetails', {}).get('shipper', {}).get('telephone', '')}</p>
        </div>
        <div class="bl-section">
            <h3>CONSIGNEE (NAME AND FULL ADDRESS)</h3>
            <p>{doc.get('partyDetails', {}).get('consignee', {}).get('name', '')}</p>
            <p>{doc.get('partyDetails', {}).get('consignee', {}).get('address', '')}</p>
            <p>Tel: {doc.get('partyDetails', {}).get('consignee', {}).get('telephone', '')}</p>
        </div>
        <div class="bl-section">
            <h3>NOTIFY PARTY (NAME AND ADDRESS)</h3>
            <p>{doc.get('partyDetails', {}).get('notifyParty', {}).get('name', '')}</p>
            <p>{doc.get('partyDetails', {}).get('notifyParty', {}).get('address', '')}</p>
            <p>Tel: {doc.get('partyDetails', {}).get('notifyParty', {}).get('telephone', '')}</p>
        </div>
        <div class="bl-grid">
            <div class="bl-section">
                <h3>PLACE OF RECEIPT BY PRE-CARRIER</h3>
                <p>{doc.get('routeDetails', {}).get('placeOfReceipt', '')}</p>
            </div>
            <div class="bl-section">
                <h3>PORT OF LOADING</h3>
                <p>{doc.get('routeDetails', {}).get('portOfLoading', '')}</p>
            </div>
        </div>
        <div class="bl-grid">
            <div class="bl-section">
                <h3>VESSEL</h3>
                <p>{doc.get('voyageDetails', {}).get('vesselName', '')}</p>
            </div>
            <div class="bl-section">
                <h3>PORT OF DISCHARGE</h3>
                <p>{doc.get('routeDetails', {}).get('portOfDischarge', '')}</p>
            </div>
        </div>
    <div class="bl-section">
        <h3>PARTICULARS FURNISHED BY SHIPPER - CARRIER NOT RESPONSIBLE</h3>
        {container_info_html}
    </div>
    <div class="bl-footer">
        <div class="bl-grid">
            <div>
                <p><strong>FCL/FCL</strong></p>
                <p><strong>{doc['shippingTerm']}</strong></p>
            </div>
            <div>
                <p><strong>"{doc['paymentDetails']['freightPaymentTerms'].upper()}"</strong></p>
                <p>{doc['totalShipment']['totalContainers']}</p>
            </div>
        </div>
        <p><strong>TOTAL No. OF CONTAINERS OF PACKAGES RECEIVED BY THE CARRIER: {doc['totalShipment']['totalContainers']}</strong></p>
        <p class="small-text">The number of containers of packages shown in the 'TOTAL No. OF CONTAINERS OR PACKAGES RECEIVED BY THE CARRIER'S box which are said by the shipper to hold or consolidate the goods described in the PARTICULARS FURNISHED BY SHIPPER - CARRIER NOT RESPONSIBLE box, have been received by Sea Lead Shipping DMCC from the shipper in apparent good order and condition except as otherwise indicated hereon - weight, measure, marks, numbers, quality, quantity, description, contents and value unknown - for Carriage from the Place of Receipt or the Port of loading (whichever is applicable) to the Port of Discharge or the Place of Delivery (whichever is applicable) on the terms and conditions hereof INCLUDING THE TERMS AND CONDITIONS ON THE REVERSE SIDE HEREOF, THE CARRIER'S APPLICABLE TARIFF AND THE TERMS AND CONDITIONS OF THE PRECARRIER AND ONCARRIER AS APPLICABLE IN ACCORDANCE WITH THE TERMS AND CONDITIONS ON THE REVERSE SIDE HEREOF.</p>
        <p class="small-text">IN WITNESS WHEREOF {doc['documentationDetails']['numberOfOriginalBLs']} ({doc['documentationDetails']['numberOfOriginalBLs']} in words) ORIGINAL BILLS OF LADING (unless otherwise stated above) HAVE BEEN SIGNED ALL OF THE SAME TENOR AND DATE, ONE OF WHICH BEING ACCOMPLISHED THE OTHER(S) TO STAND VOID.</p>
        <div class="bl-grid">
            <div>
                <p><strong>CHERRY SHIPPING LINE</strong></p>
                <p><strong>as Carrier</strong></p>
                <p>By ContainerGenie.ai CO., LTD.</p>
                <p>as Agents only for Carrier</p>
            </div>
            <div>
                <p><strong>Place Issued: {doc['paymentDetails']['freightPayableAt']}</strong></p>
                <p><strong>Date Issued: {doc['additionalInformation']['onboardDate']}</strong></p>
            </div>
        </div>
    </div>
    """
    
    # Render the BL form
    st.markdown(bl_html, unsafe_allow_html=True)


def generate_container_rows(containers, doc):
    table_html = f"""
    <table class="bl-table">
        <tr>
            <th>MARKS AND NUMBERS</th>
            <th>NO. OF PKGS.</th>
            <th>DESCRIPTION OF PACKAGES AND GOODS</th>
            <th>GROSS WEIGHT (KG)</th>
            <th>MEASUREMENT (CBM)</th>
        </tr>
        <tr>
            <td>{containers[0]['marksAndNumbers']}</td>
            <td>{doc['totalShipment']['totalPackages']} {containers[0]['packageType']}</td>
            <td>{doc['commodityDescription']}</td>
            <td>{doc['totalShipment']['totalGrossWeight']}</td>
            <td>{doc['totalShipment']['totalMeasurement']}</td>
        </tr>
    </table>
    <h3>CONTAINER INFORMATION</h3>
    <table class="bl-table">
        <tr>
            <th>Container No.</th>
            <th>Seal No.</th>
            <th>No. of Pkgs</th>
            <th>Description</th>
            <th>Gross Weight (KG)</th>
            <th>Measurement (CBM)</th>
        </tr>
    """
    
    for container in containers:
        table_html += f"""
        <tr>
            <td>{container.get('containerNumber', '')}</td>
            <td>{container.get('sealNumber', '')}</td>
            <td>{container.get('numberOfPackages', '')} {container.get('packageType', '')}</td>
            <td>SHIPPER'S LOAD, COUNT & WEIGHT, SOTW & SEAL SAID TO CONTAIN: {container.get('cargoDescription', '')}</td>
            <td>{container.get('grossWeight', ''):.3f}</td>
            <td>{container.get('measurement', ''):.4f}</td>
        </tr>
        """
    table_html += "</table>"
    return table_html





def main():
    st.title("BL Viewer")

    # Fetch all documents from MongoDB
    documents = list(collection.find())

    # Create a list of booking references
    booking_refs = [doc.get('bookingReference', 'Unknown') for doc in documents]
    
    # Create a selectbox for choosing a document by booking reference
    selected_booking_ref = st.selectbox(
        "Select a BL document",
        options=booking_refs,
        format_func=lambda x: f"Booking Ref: {x}"
    )

    # Find the selected document
    selected_doc = next((doc for doc in documents if doc.get('bookingReference') == selected_booking_ref), None)

    # Display the selected document as a BL form
    if selected_doc:
        display_bl_form(selected_doc)
    else:
        st.warning("No document found for the selected booking reference.")

if __name__ == "__main__":
    main()