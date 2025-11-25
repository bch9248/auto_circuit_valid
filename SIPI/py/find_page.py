import os
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from pathlib import Path

def extract_cpu11_page():
    # Define paths
    pdf_path = "data/G2i_MerinoW1416_SI_0716 1.pdf"
    target_directory = "extracted_pages"
    
    # Create target directory if it doesn't exist
    os.makedirs(target_directory, exist_ok=True)
    
    # Initialize document converter
    converter = DocumentConverter()
    
    try:
        # Convert the PDF
        print(f"Processing PDF: {pdf_path}")
        result = converter.convert(pdf_path)
        
        # Get the document
        doc = result.document
        
        # Search for CPU11 in the document
        cpu11_page = None
        cpu11_content = None
        
        # Method 1: Search through pages for CPU11 content
        for page_num, page in enumerate(doc.pages, 1):
            page_text = page.text if hasattr(page, 'text') else str(page)
            
            # Look for CPU11 reference
            if "CPU11" in page_text.upper() or "CPU 11" in page_text.upper():
                cpu11_page = page_num
                cpu11_content = page_text
                print(f"Found CPU11 content on page {page_num}")
                break
        
        # Method 2: If not found by content search, use table of contents info
        # From the image, CPU11 appears to be item 39 in the TOC
        if cpu11_page is None:
            # Based on typical document structure, estimate page number
            # You may need to adjust this based on the actual document structure
            estimated_page = 39  # This might need adjustment
            if estimated_page <= len(doc.pages):
                cpu11_page = estimated_page
                cpu11_content = doc.pages[estimated_page - 1].text if hasattr(doc.pages[estimated_page - 1], 'text') else str(doc.pages[estimated_page - 1])
                print(f"Using estimated page {estimated_page} for CPU11")
        
        if cpu11_page and cpu11_content:
            # Save the extracted page content
            output_file = os.path.join(target_directory, f"CPU11_page_{cpu11_page}.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"CPU11 Content - Page {cpu11_page}\n")
                f.write("=" * 50 + "\n\n")
                f.write(cpu11_content)
            
            print(f"CPU11 content extracted to: {output_file}")
            
            # Also save as markdown for better formatting
            md_output_file = os.path.join(target_directory, f"CPU11_page_{cpu11_page}.md")
            with open(md_output_file, 'w', encoding='utf-8') as f:
                f.write(f"# CPU11 Content - Page {cpu11_page}\n\n")
                f.write(cpu11_content)
            
            print(f"CPU11 content also saved as markdown: {md_output_file}")
            
            return cpu11_page, output_file
        else:
            print("CPU11 content not found in the document")
            return None, None
            
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None, None

def search_table_of_contents():
    """Alternative method to parse TOC and find CPU11 page number"""
    pdf_path = "data/G2i_MerinoW1416_SI_0716 1.pdf"
    converter = DocumentConverter()
    
    # try:
    result = converter.convert(pdf_path)
    doc = result.document
    
    # Search for table of contents
    for page_num, page in enumerate(doc.pages, 1):
        page_text = page.text if hasattr(page, 'text') else str(page)
        
        # if "CPU11" in page_text and "TABLE OF CONTENTS" in page_text.upper():
        if "TABLE OF CONTENTS" in page_text.upper():
            # print(f"Found TOC with CPU11 on page {page_num}")
            print(f"Found TOC on page {page_num}")
            
            # Extract the line containing CPU11
            lines = page_text.split('\n')
            for line in lines:
                if "WWAN" in line:
                    print(f"WWAN entry: {line.strip()}")
                    # Try to extract page number from the line
                    # This will depend on the exact format of your TOC
                    parts = line.split()
                    for part in parts:
                        if part.isdigit():
                            cpu11_page_num = int(part)
                            print(f"WWAN appears to be on page: {cpu11_page_num}")
                            return cpu11_page_num
            break
        
        return None


if __name__ == "__main__":
    # First try to find the exact page number from TOC
    toc_page = search_table_of_contents()
    
    if toc_page:
        print(f"TOC indicates CPU11 is on page {toc_page}")
    
    # # Extract the CPU11 page
    # page_num, output_file = extract_cpu11_page()
    
    # if page_num and output_file:
    #     print(f"Successfully extracted CPU11 content from page {page_num}")
    #     print(f"Output saved to: {output_file}")
    # else:
    #     print("Failed to extract CPU11 content")