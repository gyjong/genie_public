import streamlit as st
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import base64
import html

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

def escape_markdown(text):
    return html.escape(str(text)).replace("|", "\\|").replace("\n", "<br>")

def generate_container_rows(containers, doc):
    table_md = f"""
| MARKS AND NUMBERS | NO. OF PKGS. | DESCRIPTION OF PACKAGES AND GOODS | GROSS WEIGHT (KG) | MEASUREMENT (CBM) |
|-------------------|--------------|-----------------------------------|-------------------|-------------------|
| {escape_markdown(containers[0]['marksAndNumbers'])} | {escape_markdown(doc['totalShipment']['totalPackages'])} {escape_markdown(containers[0]['packageType'])} | {escape_markdown(doc['commodityDescription'])} | {escape_markdown(doc['totalShipment']['totalGrossWeight'])} | {escape_markdown(doc['totalShipment']['totalMeasurement'])} |

### CONTAINER INFORMATION

| Container No. | Seal No. | No. of Pkgs | Description | Gross Weight (KG) | Measurement (CBM) |
|---------------|----------|-------------|-------------|-------------------|-------------------|
"""
    
    for container in containers:
        try:
            formatted_weight = f"{container['grossWeight']:.3f}"
            formatted_measurement = f"{container['measurement']:.4f}"
            table_md += f"| {escape_markdown(container.get('containerNumber', ''))} | {escape_markdown(container.get('sealNumber', ''))} | {escape_markdown(container.get('numberOfPackages', 0))} {escape_markdown(container.get('packageType', ''))} | {escape_markdown(container.get('cargoDescription', ''))} | {formatted_weight} | {formatted_measurement} |\n"
        except KeyError as e:
            print(f"Error generating container row: {e}")
    
    return table_md

def display_bl_form(doc):
    # Apply custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)
    
    # Load and encode the logo
    logo_base64 = get_base64_encoded_image("./img/containergenie.png")
    
    # Generate container information
    container_info_md = generate_container_rows(doc['containers'], doc)

    # Create the BL form using Markdown
    bl_md = f"""
# BILL OF LADING

**BOOKING NO:** {escape_markdown(doc.get('bookingReference', ''))}  
**SERVICE:** {escape_markdown(doc.get('service', ''))}  
**BL NO:** {escape_markdown(doc.get('bookingReference', ''))}

### SHIPPER (NAME AND FULL ADDRESS)
{escape_markdown(doc.get('partyDetails', {}).get('shipper', {}).get('name', ''))}  
{escape_markdown(doc.get('partyDetails', {}).get('shipper', {}).get('address', ''))}  
Tel: {escape_markdown(doc.get('partyDetails', {}).get('shipper', {}).get('telephone', ''))}

### CONSIGNEE (NAME AND FULL ADDRESS)
{escape_markdown(doc.get('partyDetails', {}).get('consignee', {}).get('name', ''))}  
{escape_markdown(doc.get('partyDetails', {}).get('consignee', {}).get('address', ''))}  
Tel: {escape_markdown(doc.get('partyDetails', {}).get('consignee', {}).get('telephone', ''))}

### NOTIFY PARTY (NAME AND ADDRESS)
{escape_markdown(doc.get('partyDetails', {}).get('notifyParty', {}).get('name', ''))}  
{escape_markdown(doc.get('partyDetails', {}).get('notifyParty', {}).get('address', ''))}  
Tel: {escape_markdown(doc.get('partyDetails', {}).get('notifyParty', {}).get('telephone', ''))}

| PLACE OF RECEIPT BY PRE-CARRIER | PORT OF LOADING |
|----------------------------------|-----------------|
| {escape_markdown(doc.get('routeDetails', {}).get('placeOfReceipt', ''))} | {escape_markdown(doc.get('routeDetails', {}).get('portOfLoading', ''))} |

| VESSEL | PORT OF DISCHARGE |
|--------|-------------------|
| {escape_markdown(doc.get('voyageDetails', {}).get('vesselName', ''))} | {escape_markdown(doc.get('routeDetails', {}).get('portOfDischarge', ''))} |

### PARTICULARS FURNISHED BY SHIPPER - CARRIER NOT RESPONSIBLE

{container_info_md}

**FCL/FCL**  
**{escape_markdown(doc['shippingTerm'])}**

**"{escape_markdown(doc['paymentDetails']['freightPaymentTerms'].upper())}"**  
{escape_markdown(doc['totalShipment']['totalContainers'])}

**TOTAL No. OF CONTAINERS OF PACKAGES RECEIVED BY THE CARRIER: {escape_markdown(doc['totalShipment']['totalContainers'])}**

The number of containers of packages shown in the 'TOTAL No. OF CONTAINERS OR PACKAGES RECEIVED BY THE CARRIER'S box which are said by the shipper to hold or consolidate the goods described in the PARTICULARS FURNISHED BY SHIPPER - CARRIER NOT RESPONSIBLE box, have been received by Sea Lead Shipping DMCC from the shipper in apparent good order and condition except as otherwise indicated hereon - weight, measure, marks, numbers, quality, quantity, description, contents and value unknown - for Carriage from the Place of Receipt or the Port of loading (whichever is applicable) to the Port of Discharge or the Place of Delivery (whichever is applicable) on the terms and conditions hereof INCLUDING THE TERMS AND CONDITIONS ON THE REVERSE SIDE HEREOF, THE CARRIER'S APPLICABLE TARIFF AND THE TERMS AND CONDITIONS OF THE PRECARRIER AND ONCARRIER AS APPLICABLE IN ACCORDANCE WITH THE TERMS AND CONDITIONS ON THE REVERSE SIDE HEREOF.

IN WITNESS WHEREOF {escape_markdown(doc['documentationDetails']['numberOfOriginalBLs'])} ({escape_markdown(doc['documentationDetails']['numberOfOriginalBLs'])} in words) ORIGINAL BILLS OF LADING (unless otherwise stated above) HAVE BEEN SIGNED ALL OF THE SAME TENOR AND DATE, ONE OF WHICH BEING ACCOMPLISHED THE OTHER(S) TO STAND VOID.

**CHERRY SHIPPING LINE**  
**as Carrier**  
By ContainerGenie.ai CO., LTD.  
as Agents only for Carrier

**Place Issued: {escape_markdown(doc['paymentDetails']['freightPayableAt'])}**  
**Date Issued: {escape_markdown(doc['additionalInformation']['onboardDate'])}**
    """
    
    # Render the BL form
    st.markdown(bl_md)

    # Display the logo separately
    st.image(f"data:image/png;base64,{logo_base64}", width=250)

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