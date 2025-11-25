# Use sliding window to find image and crop

import requests            # For making HTTP POST requests to Azure/OpenAI or Local API
import time                # For sleep (retry delays)
import os
from langchain.schema import BaseMessage, AIMessage  # LangChain message schema
import requests
import time
from langchain.schema import BaseMessage, HumanMessage, AIMessage

import base64
import json
from PIL import Image
import numpy as np

from Agent.Agent import AzureChat



def encode_image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def crop_image_sliding_window(image_path, window_size=(800, 600), overlap=0.2):
    """
    Crop image using sliding window approach
    
    Args:
        image_path: Path to input image
        window_size: Tuple of (width, height) for crop window
        overlap: Overlap ratio between windows (0.0 to 1.0)
    
    Returns:
        List of tuples: (cropped_image_path, x, y, width, height, four_corners)
    """
    image = Image.open(image_path)
    img_width, img_height = image.size
    
    window_width, window_height = window_size
    step_x = int(window_width * (1 - overlap))
    step_y = int(window_height * (1 - overlap))
    
    cropped_images = []
    crop_counter = 0
    
    # Create output directory for cropped images
    output_dir = "cropped_windows"
    os.makedirs(output_dir, exist_ok=True)
    
    for y in range(0, img_height - window_height + 1, step_y):
        for x in range(0, img_width - window_width + 1, step_x):
            # Crop the image
            crop_box = (x, y, x + window_width, y + window_height)
            cropped_img = image.crop(crop_box)
            
            # Save cropped image
            crop_filename = f"crop_{crop_counter:04d}_x{x}_y{y}.png"
            crop_path = os.path.join(output_dir, crop_filename)
            cropped_img.save(crop_path)
            
            # Calculate four corners of the patch
            four_corners = [
                (x, y),  # Top-left
                (x + window_width, y),  # Top-right
                (x + window_width, y + window_height),  # Bottom-right
                (x, y + window_height)  # Bottom-left
            ]
            
            cropped_images.append({
                'path': crop_path,
                'x': x,
                'y': y,
                'width': window_width,
                'height': window_height,
                'crop_id': crop_counter,
                'four_corners': four_corners,
                'bounding_box': (x, y, x + window_width, y + window_height)
            })
            
            crop_counter += 1
    
    return cropped_images


def check_text_in_image(chat_model, image_path, target_texts):
    """
    Use VLM to check if target texts are present in the image
    
    Args:
        chat_model: The vision language model
        image_path: Path to the image
        target_texts: List of texts to search for
    
    Returns:
        Dict with confidence scores and detected texts
    """
    if not target_texts:
        return {
            'confidence': 0,
            'detections': {},
            'explanation': "No target texts provided",
            'detected_count': 0
        }
    
    target_text_str = ", ".join(target_texts)
    formatted_text = "\n".join([f"- {txt}: [Yes/No]" for txt in target_texts])
    
    # Check for different component types
    ground_terms = ['GND', 'GROUND', 'DGND', 'AGND', 'PGND', 'SGND', 'CHASSIS', 'EARTH', 'ground', 'gnd']
    capacitor_terms = ['uF', 'nF', 'pF', 'F', 'capacitor', 'cap', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    resistor_terms = ['ohm', 'Ω', 'R1', 'R2', 'R3', 'R4', 'R5', 'resistor', 'kΩ', 'MΩ']
    
    contains_ground = any(any(term.lower() in text.lower() for term in ground_terms) for text in target_texts)
    contains_capacitor = any(any(term in text for term in capacitor_terms) for text in target_texts)
    contains_resistor = any(any(term in text for term in resistor_terms) for text in target_texts)
    
    # Enhanced prompt based on component types
    if contains_ground or contains_capacitor or contains_resistor:
        component_instructions = []
        
        if contains_ground:
            component_instructions.append("""- GROUND terms (GND, GROUND, etc.): Detect BOTH text labels AND ground symbols (⏚ - three horizontal lines stacked vertically)""")
        
        if contains_capacitor:
            component_instructions.append("""- CAPACITOR values (uF, nF, pF, etc.): Look for capacitor symbols (two parallel lines or curved line + straight line) AND their associated values like "4.7uF", "220nF", "100pF", etc.""")
        
        if contains_resistor:
            component_instructions.append("""- RESISTOR values: Look for resistor symbols (zigzag or rectangular) AND their values like "10kΩ", "1MΩ", etc.""")
        
        instructions_text = "\n".join(component_instructions)
        
        query = f"""Analyze this electronic schematic image and determine if ANY of the following components/texts are clearly visible: {target_text_str}

IMPORTANT DETECTION RULES:
{instructions_text}
- TEXT labels: Look for exact text matches (case-insensitive)
- COMPONENT symbols: Identify the schematic symbols and their associated values

Please respond with:
1. List each item and whether it's detected (Yes/No)
2. Brief explanation of what components and values you see

Format your response EXACTLY as:
DETECTIONS:
{formatted_text}
EXPLANATION: [brief description including component symbols and values you see]

Be precise with the format - use exactly "Yes" or "No" for each detection.
For component values, answer "Yes" if you find EITHER the exact value text OR a capacitor/resistor symbol with that value nearby."""
    else:
        query = f"""Analyze this image and determine if ANY of the following texts are clearly visible: {target_text_str}

Please respond with:
1. List each text and whether it's detected (Yes/No)
2. Brief explanation of what you see

Format your response EXACTLY as:
DETECTIONS:
{formatted_text}
EXPLANATION: [brief description]

Be precise with the format - use exactly "Yes" or "No" for each detection."""

    input_img_base64=encode_image_to_base64(image_path)
    messages = [HumanMessage(content=[
        {
            "type": "text",
            "text": query
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{input_img_base64}"
            }
        }
    ])]
    
    try:
        response = chat_model(messages)
        return parse_detection_response(response.content, target_texts)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return {
            'confidence': 0,
            'detections': {text: False for text in target_texts},
            'explanation': f"Error: {str(e)}",
            'detected_count': 0
        }

def parse_detection_response(response_text, target_texts):
    """Parse the VLM response to extract confidence and detections"""
    lines = response_text.strip().split('\n')
    
    detections = {}
    explanation = ""
    
    # Initialize all target texts as not detected
    for text in target_texts:
        detections[text] = False
    
    # Parse the response
    in_detections_section = False
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('DETECTIONS:'):
            in_detections_section = True
            continue
        elif line.startswith('EXPLANATION:'):
            in_detections_section = False
            explanation = line.split(':', 1)[1].strip() if ':' in line else ""
            continue
        
        if in_detections_section and line.startswith('- '):
            try:
                # Remove the "- " prefix and split on ":"
                content = line[2:].strip()
                if ':' in content:
                    text_part, detection_part = content.split(':', 1)
                    text = text_part.strip()
                    detected_str = detection_part.strip().lower()
                    
                    # Check if this text is in our target list (case-insensitive matching)
                    matching_target = None
                    for target in target_texts:
                        if target.lower() == text.lower():
                            matching_target = target
                            break
                    
                    if matching_target:
                        detected = detected_str in ['yes', 'true', '[yes]', 'detected']
                        detections[matching_target] = detected
            except Exception as e:
                print(f"Warning: Could not parse detection line: '{line}' - {e}")
                continue
    
    # Count detected strings and calculate confidence as percentage
    detected_count = sum(1 for detected in detections.values() if detected)
    confidence = (detected_count / len(target_texts)) * 100 if target_texts else 0
    
    return {
        'confidence': confidence,
        'detections': detections,
        'explanation': explanation,
        'detected_count': detected_count,
        'raw_response': response_text
    }

def crop_image_with_coordinates(input_path, output_path, pin_coordinates, padding=50):
    """
    Crop an image to include all specified pin coordinates with padding.
    
    Args:
        input_path (str): Path to the input image
        output_path (str): Path to save the cropped image
        pin_coordinates (list): List of (x, y) coordinates
        padding (int): Additional padding around the bounding box
    """
    # Open the image
    image = Image.open(input_path)
    
    # Extract x and y coordinates
    x_coords = [coord[0] for coord in pin_coordinates]
    y_coords = [coord[1] for coord in pin_coordinates]
    
    # Find bounding box
    min_x = min(x_coords) - padding
    min_y = min(y_coords) - padding
    max_x = max(x_coords) + padding
    max_y = max(y_coords) + padding
    
    # Ensure coordinates are within image bounds
    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(image.width, max_x)
    max_y = min(image.height, max_y)
    
    # Crop the image using the bounding box
    cropped_image = image.crop((min_x, min_y, max_x, max_y))
    
    # Save the cropped image
    cropped_image.save(output_path)
    
    print(f"Cropped area: ({min_x}, {min_y}, {max_x}, {max_y})")
    print(f"Original size: {image.width}x{image.height}")
    print(f"Cropped size: {cropped_image.width}x{cropped_image.height}")
    
    return cropped_image

def gather_patch_coordinates(results, min_detections=2):
    """
    Gather coordinates from patches that detected more than specified number of strings
    
    Args:
        results: List of detection results
        min_detections: Minimum number of detected strings required
    
    Returns:
        List of all coordinates from qualifying patches
    """
    all_coordinates = []
    qualifying_patches = []
    
    for result in results:
        detected_count = result['detection_result']['detected_count']
        if detected_count >= min_detections:
            # Add all four corners of this patch
            corners = result['crop_info']['four_corners']
            all_coordinates.extend(corners)
            qualifying_patches.append(result)
    
    return all_coordinates, qualifying_patches
def validate_target_texts(target_texts):
    """Validate and clean target texts input"""
    if isinstance(target_texts, str):
        # If single string, convert to list
        target_texts = [target_texts]
    elif not isinstance(target_texts, list):
        raise ValueError("target_texts must be a string or list of strings")
    
    # Remove empty strings and strip whitespace
    cleaned_texts = [text.strip() for text in target_texts if text and text.strip()]
    
    if not cleaned_texts:
        raise ValueError("No valid target texts provided")
    
    return cleaned_texts

def process_image_with_targets(chat_model, input_image, target_texts, window_size, overlap, min_strings_threshold, query_number):
    """
    Process a single image with a specific set of target texts
    
    Args:
        chat_model: The vision language model
        input_image: Path to input image
        target_texts: List of target texts to search for
        window_size: Tuple of (width, height) for crop window
        overlap: Overlap ratio between windows
        min_strings_threshold: Minimum number of detected strings per patch
        query_number: Query number for output organization
    
    Returns:
        Dict containing all processing results
    """
    print(f"\n{'='*60}")
    print(f"QUERY {query_number}: Processing with target texts:")
    print(f"  {target_texts}")
    print(f"  Window size: {window_size}, Overlap: {overlap}")
    print(f"  Minimum strings threshold: {min_strings_threshold}")
    
    # Step 1: Crop image using sliding window
    print(f"\nStep 1: Cropping image with sliding window...")
    cropped_images = crop_image_sliding_window(input_image, window_size, overlap)
    print(f"  Generated {len(cropped_images)} cropped images")
    
    # Step 2: Process each cropped image with VLM
    print(f"\nStep 2: Processing cropped images with VLM...")
    results = []
    
    for i, crop_info in enumerate(cropped_images):
        print(f"  Processing crop {i+1}/{len(cropped_images)}: {os.path.basename(crop_info['path'])}")
        
        detection_result = check_text_in_image(chat_model, crop_info['path'], target_texts)
        
        result = {
            'crop_info': crop_info,
            'detection_result': detection_result
        }
        results.append(result)
        
        # Print progress
        confidence = detection_result['confidence']
        detected_count = detection_result['detected_count']
        print(f"    Confidence: {confidence:.1f}%, Detected strings: {detected_count}/{len(target_texts)}")
    
    # Step 3: Gather coordinates from patches with multiple detections
    print(f"\nStep 3: Gathering patches with {min_strings_threshold}+ detected strings...")
    all_coordinates, qualifying_patches = gather_patch_coordinates(results, min_strings_threshold)
    
    print(f"  Found {len(qualifying_patches)} patches with {min_strings_threshold}+ detected strings")
    print(f"  Total coordinates collected: {len(all_coordinates)}")
    
    # Step 4: Create final cropped image if we have qualifying patches
    final_cropped_path = None
    if all_coordinates:
        query_output_dir = os.path.join("Output", f"query_{query_number}")
        os.makedirs(query_output_dir, exist_ok=True)
        
        final_cropped_path = os.path.join(query_output_dir, "final_cropped_region.png")
        
        print(f"\nStep 4: Creating final cropped image...")
        final_cropped = crop_image_with_coordinates(
            input_image, 
            final_cropped_path, 
            all_coordinates, 
            padding=50
        )
        print(f"  Final cropped image saved to: {final_cropped_path}")
    else:
        print(f"\nStep 4: No patches found with {min_strings_threshold}+ detected strings")
    
    return {
        'query_number': query_number,
        'target_texts': target_texts,
        'window_size': window_size,
        'overlap': overlap,
        'min_strings_threshold': min_strings_threshold,
        'cropped_images': cropped_images,
        'results': results,
        'qualifying_patches': qualifying_patches,
        'all_coordinates': all_coordinates,
        'final_cropped_path': final_cropped_path,
        'stats': {
            'total_crops': len(cropped_images),
            'qualifying_patches_count': len(qualifying_patches),
            'max_confidence': max(r['detection_result']['confidence'] for r in qualifying_patches) if qualifying_patches else 0,
            'max_detections': max(r['detection_result']['detected_count'] for r in qualifying_patches) if qualifying_patches else 0
        }
    }

def save_query_results(query_result, input_image):
    """
    Save results for a single query to organized output directories
    
    Args:
        query_result: Result dictionary from process_image_with_targets
        input_image: Path to input image
    """
    query_number = query_result['query_number']
    query_output_dir = os.path.join("Output", f"query_{query_number}")
    os.makedirs(query_output_dir, exist_ok=True)
    
    # Save detailed results
    detailed_results_path = os.path.join(query_output_dir, "text_detection_results.json")
    with open(detailed_results_path, "w") as json_file:
        # Create a serializable version of the results
        serializable_results = []
        for result in query_result['results']:
            serializable_result = {
                'crop_info': result['crop_info'],
                'detection_result': result['detection_result']
            }
            serializable_results.append(serializable_result)
        
        json.dump({
            'input_image': input_image,
            'query_number': query_number,
            'target_texts': query_result['target_texts'],
            'window_size': query_result['window_size'],
            'overlap': query_result['overlap'],
            'total_crops': query_result['stats']['total_crops'],
            'qualifying_patches': query_result['stats']['qualifying_patches_count'],
            'min_strings_threshold': query_result['min_strings_threshold'],
            'all_results': serializable_results
        }, json_file, indent=4)
    
    # Save summary of qualifying patches
    summary_path = os.path.join(query_output_dir, "qualifying_patches_summary.json")
    with open(summary_path, "w") as json_file:
        json.dump({
            'query_number': query_number,
            'target_texts': query_result['target_texts'],
            'qualifying_patches': [
                {
                    'image_path': r['crop_info']['path'],
                    'position': (r['crop_info']['x'], r['crop_info']['y']),
                    'four_corners': r['crop_info']['four_corners'],
                    'bounding_box': r['crop_info']['bounding_box'],
                    'confidence': r['detection_result']['confidence'],
                    'detected_count': r['detection_result']['detected_count'],
                    'detections': r['detection_result']['detections'],
                    'explanation': r['detection_result']['explanation']
                }
                for r in query_result['qualifying_patches']
            ],
            'final_bounding_coordinates': query_result['all_coordinates'] if query_result['all_coordinates'] else None
        }, json_file, indent=4)
    
    return detailed_results_path, summary_path

def print_query_summary(query_result):
    """
    Print summary for a single query
    
    Args:
        query_result: Result dictionary from process_image_with_targets
    """
    query_number = query_result['query_number']
    stats = query_result['stats']
    
    print(f"\nQUERY {query_number} SUMMARY:")
    print(f"  Target texts: {query_result['target_texts']}")
    print(f"  Total crops processed: {stats['total_crops']}")
    print(f"  Patches with {query_result['min_strings_threshold']}+ strings: {stats['qualifying_patches_count']}")
    
    if stats['qualifying_patches_count'] > 0:
        print(f"  Best confidence score: {stats['max_confidence']:.1f}%")
        print(f"  Maximum strings detected in single patch: {stats['max_detections']}")
        if query_result['final_cropped_path']:
            print(f"  Final cropped region: {query_result['final_cropped_path']}")
    else:
        print(f"  No qualifying patches found")

def print_overall_summary(all_query_results):
    """
    Print overall summary for all queries
    
    Args:
        all_query_results: List of query result dictionaries
    """
    print(f"\n{'='*60}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*60}")
    
    total_crops = sum(r['stats']['total_crops'] for r in all_query_results)
    total_qualifying = sum(r['stats']['qualifying_patches_count'] for r in all_query_results)
    successful_queries = sum(1 for r in all_query_results if r['stats']['qualifying_patches_count'] > 0)
    
    print(f"Total queries processed: {len(all_query_results)}")
    print(f"Successful queries (with qualifying patches): {successful_queries}")
    print(f"Total crops across all queries: {total_crops}")
    print(f"Total qualifying patches across all queries: {total_qualifying}")
    
    print(f"\nResults saved to:")
    for result in all_query_results:
        query_num = result['query_number']
        print(f"  Query {query_num}: ./Output/query_{query_num}/")
    print(f"  Cropped images: ./cropped_windows/")

if __name__ == "__main__":
    # Configuration - Multiple target_texts sets
    input_image = "Input/page_87_300dpi.png"
    
    # Define multiple sets of target texts
    target_texts_sets = {
        # -1:["PVSIM", "DGND", "0.1uF","18pF"],
        # 0:["3P3VAUX_2", "3P3VAUX_4", "3P3VAUX_70", "3P3VAUX_72", "3P3VAUX_74", "PVSIM"],  # Combined set
        # 1:["PVSIM","PAD1400","P3V3DS"],
        # 2:["UIM_CD","PVSIM","WWAN_UIM_CD#"],
        # 3:["SIM DETECT", "UIM_CD"],
        # 4:["WWAN_UIM_RST", "WWAN_UIM_CLK","WWAN_UIM_DATA","UIM-RESET","UIM-CLK","UIM-DATA"],
        # 5:["WWAN_UIM_RST", "WWAN_UIM_CLK","WWAN_UIM_DATA","UIM_RST_CN","UIM_CLK_CN","UIM_DATA_CN"],
    #     6: [
    #     "UIM PWR CN",
    #     "UIM RST CN",
    #     "UIM CLK CN",
    #     "UIM PWR CN",
    #     "R1426 1 2 100K 5% 1 0.4MM NI",
    #     "UIM CD# CN",
    #     "UIM VPP CN",
    #     "UIM DATA CN",
    #     "UIM PWR CN",
    #     "R1427 1 2 20K 5% 1 0.4MM NI",
    #     "Capacitors",
    #     "Component Value Voltage Spacing",
    #     "C1454 0.1uF 16V 0.65MM",
    #     "C1453 4.7uF 6.3V 0.75MM",
    #     "C1452 18PF 50V 0.65MM",
    #     "C1451 18PF 50V 0.65MM",
    #     "C1450 18PF 50V 0.43MM",
    #     "CAP CLOSE TO SIM CARD",
    #     "Connector CN1400",
    #     "Pin Signal Pin Signal",
    #     "C1 VCC G1",
    #     "C2 RST G2",
    #     "C3 CLK G3",
    #     "C5 GND G4",
    #     "C6 CD/SW G5",
    #     "C6 VPP G6",
    #     "C7 DATA G7",
    #     "G8",
    #     "FULLINHOPE_A1012_L001_7P",
    #     "6026B0588001_TK2",
    #     "2.32MM",
    # ]
        7:['COEX3', '60', 'COEX2', '62', 'COEX1', '64', '66']
    }

    temp_image = Image.open(input_image)
    img_width, img_height = temp_image.size

    window_size = (1400, 400)  # One fourth of the whole resolution
    # window_size = (img_width // 4, img_height // 4)  # One fourth of the whole resolution
    overlap = 0.3  # 30% overlap
    MIN_STR_THRESHOLD = 2  # Minimum number of detected strings per patch
    
    # Initialize the chat model
    # chat_model = AzureChat(deployment_url=os.getenv("OPENAI_API_KEY_GPT4_1"))
    # model_name="gpt-4.1"
    # temperature=0
    # max_tokens=13107
    # deployment_name = "gpt-4.1-mini"  # or your own deployment
#     deployment_name = "gpt-35-turbo"  # or your own deployment
    openai_api_url = os.getenv("OPENAI_API_URL_GPT4_1")
    openai_api_key = os.getenv("OPENAI_API_KEY_GPT4_1")
    chat_model = AzureChat(
        api_url=openai_api_url,
        api_key=openai_api_key,
        model_name="gpt-4.1",
        temperature=0,
        max_tokens=13107,
    )
    print(f"Processing image: {input_image}")
    print(f"Total query sets: {len(target_texts_sets)}")
    print(f"Window size: {window_size}, Overlap: {overlap}")
    
    
    # Create main output directory
    os.makedirs("Output", exist_ok=True)
    
    # Process each set of target texts
    all_query_results = []
    
    for query_num, target_texts in target_texts_sets.items():
        # min_strings_threshold=MIN_STR_THRESHOLD if len(target_texts)>2 else 1
        min_strings_threshold=len(target_texts)-2
        print(f"Minimum strings threshold: {min_strings_threshold}")
        # Validate and clean target texts
        target_texts = validate_target_texts(target_texts)
        
        # Process this query
        query_result = process_image_with_targets(
            chat_model, 
            input_image, 
            target_texts, 
            window_size, 
            overlap, 
            min_strings_threshold, 
            query_num
        )
        
        # Save results for this query
        detailed_path, summary_path = save_query_results(query_result, input_image)
        
        # Print summary for this query
        print_query_summary(query_result)
        
        all_query_results.append(query_result)
    
    # Print overall summary
    print_overall_summary(all_query_results)