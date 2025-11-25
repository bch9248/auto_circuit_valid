
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
    
    # Check if any target contains ground-related terms
    ground_terms = ['GND', 'GROUND', 'DGND', 'AGND', 'PGND', 'SGND', 'CHASSIS', 'EARTH', 'ground', 'gnd']
    contains_ground = any(any(term.lower() in text.lower() for term in ground_terms) for text in target_texts)
    
    # Enhanced prompt that handles both text and ground symbols
    if contains_ground:
        query = f"""Analyze this schematic image and determine if ANY of the following texts or symbols are clearly visible: {target_text_str}

IMPORTANT DETECTION RULES:
1. For TEXT labels: Look for exact text matches (case-insensitive)
2. For GROUND terms (GND, GROUND, DGND, etc.): Detect ONLY:
   - Ground SYMBOLS: The standard ground symbol (⏚) which appears as three horizontal lines stacked vertically, each line shorter than the one above it, connected to circuit traces

Please respond with:
1. List each text/symbol and whether it's detected (Yes/No)
2. Brief explanation of what you see

Format your response EXACTLY as:
DETECTIONS:
{formatted_text}
EXPLANATION: [brief description including any ground symbols you see]

Be precise with the format - use exactly "Yes" or "No" for each detection.
For ground-related items, answer "Yes" if you find EITHER the text label OR the ground symbol."""
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