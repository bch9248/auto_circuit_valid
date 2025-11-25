import os
import json
import base64
from datetime import datetime
from langchain.schema import HumanMessage
from Agent.Agent import AzureChat

def encode_image_to_base64(image_path):
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_text_from_image(image_path, chat_model):
    """Extract text from a single image using Azure GPT-4 Vision"""
    
    # Encode image to base64
    base64_image = encode_image_to_base64(image_path)
    
    # Determine image format
    image_ext = os.path.splitext(image_path)[1].lower()
    mime_type = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }.get(image_ext, 'image/jpeg')
    
    # Create message with text and image
    message_content = [
        {
            "type": "text",
            "text": """Extract all visible text from this image. 
Return only the text content as a simple list, one item per line. 
Include component labels, values, pin numbers, signals, capacitor values, resistor values, and any other readable text.
Do not add any explanations or formatting - just the raw text items."""
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_image}"
            }
        }
    ]
    
    # Create HumanMessage with the content structure
    messages = [{"role": "user", "content": message_content}]
    
    # Call the model
    response = chat_model(messages)
    
    # Parse response into list
    text_content = response.content
    text_list = [line.strip() for line in text_content.split('\n') if line.strip()]
    
    return text_list

def process_image_directory(input_dir, output_dir):
    """Process all images in directory and save results to JSON in the target text format"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Azure Chat model with vision capability
    deployment_name = "gpt-4.1-mini"  # Use gpt-4o or gpt-4o-mini for vision
    chat_model = AzureChat(deployment_name=deployment_name)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    image_files = [f for f in os.listdir(input_dir) 
                   if os.path.splitext(f.lower())[1] in image_extensions]
    
    print(f"Found {len(image_files)} images to process")
    
    target_texts_sets = {}
    total_texts = 0
    
    # Process each image
    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(input_dir, image_file)
        print(f"Processing {idx}/{len(image_files)}: {image_file}")
        
        try:
            extracted_text = extract_text_from_image(image_path, chat_model)
            
            # Use image filename (without extension) as key
            file_name_without_ext = os.path.splitext(image_file)[0]
            target_texts_sets[file_name_without_ext] = extracted_text
            total_texts += len(extracted_text)
            
            print(f"  Extracted {len(extracted_text)} text items")
            
        except Exception as e:
            print(f"  Error processing {image_file}: {str(e)}")
            file_name_without_ext = os.path.splitext(image_file)[0]
            target_texts_sets[file_name_without_ext] = []
    
    # Create the final output structure
    output_data = {
        "metadata": {
            "description": "Automatically generated target texts from images",
            "generated_by": "image text extraction script",
            "total_images": len(image_files),
            "total_texts": total_texts,
            "extraction_date": datetime.now().isoformat()
        },
        "target_texts_sets": target_texts_sets
    }
    
    # Save to JSON file
    output_file = os.path.join(output_dir, "extracted_target_texts.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Total images processed: {len(image_files)}")
    print(f"Total items extracted: {total_texts}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    # Set your input and output directories
    input_directory = "Input/SVTP803_Machu1416 CMIT_PV_CM01_picture"
    output_directory = "corpus"
    
    process_image_directory(input_directory, output_directory)