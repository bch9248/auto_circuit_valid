import requests
import time
import os
from langchain.schema import BaseMessage, AIMessage, HumanMessage
import base64
import json

def is_filtered_by_azure(response_json):
    """
    Returns True if Azure filtered the message (regardless of severity).
    """
    return response_json["choices"][0].get("finish_reason") == "content_filter"

def encode_image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class AzureChat:
    def __init__(self, deployment_name, temperature=0.7, max_tokens=1000, api_version="2025-01-01-preview"):
        if deployment_name == "gpt-35-turbo":
            self.api_url = f"https://gpt4fordg.openai.azure.com/openai/deployments/{deployment_name}/chat/completions?api-version=2025-01-01-preview"
            api_key=os.getenv("OPENAI_API_KEY_GPT35")
        elif deployment_name == "gpt-4o-mini":
            self.api_url = f"https://for-dc-test.openai.azure.com/openai/deployments/{deployment_name}/chat/completions?api-version=2025-01-01-preview"
            api_key = os.getenv("OPENAI_API_KEY_GPT4o")
        elif deployment_name == "gpt-4.1-mini":
            self.api_url = f"https://for-dc-test.openai.azure.com/openai/deployments/{deployment_name}/chat/completions?api-version=2025-01-01-preview"
            api_key = os.getenv("OPENAI_API_KEY_GPT4o")
        else:                    
            self.api_url = f"https://for-dc-test.openai.azure.com/openai/deployments/{deployment_name}/chat/completions?api-version=2024-08-01-preview"
            api_key = os.getenv("OPENAI_API_KEY_GPT4o")
        self.headers = {
            "Content-Type": "application/json",
            "api-key": api_key
        }
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(self, messages, image_path=None):
        role_map = {
            "system": "system",
            "human": "user",
            "ai": "assistant"
        }

        formatted_messages = []
        for msg in messages:
            if isinstance(msg, BaseMessage):
                message_content = {
                    "role": role_map.get(msg.type, "user"),
                }
                
                # Handle image input for vision models
                if image_path and msg.type == "human":
                    base64_image = encode_image_to_base64(image_path)
                    message_content["content"] = [
                        {
                            "type": "text",
                            "text": msg.content
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                else:
                    message_content["content"] = msg.content
                    
                formatted_messages.append(message_content)

        payload = {
            "messages": formatted_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        max_retries = 5
        delay = 5
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload)
                response.raise_for_status()        
                response_json = response.json()
                if is_filtered_by_azure(response_json):
                    print(f"Message filtered by Azure due to safety issue")
                    return AIMessage(content="unsafety_input")
                else:
                    output = response.json()["choices"][0]["message"]["content"]
                    return AIMessage(content=output)
            except Exception as e:
                print(f"[Attempt {attempt+1}] Error: {e}")
                time.sleep(delay)
                delay *= 2

        raise RuntimeError("Azure OpenAI API call failed after retries")

def query_image(image_path, question, deployment_name="gpt-4.1-mini"):
    """
    Simple function to query one image with one question
    """
    # Initialize the chat model
    chat_model = AzureChat(deployment_name=deployment_name)
    
    # Create message with the question
    messages = [HumanMessage(content=question)]
    
    # Call the model with image
    response = chat_model(messages, image_path=image_path)
    
    return response.content
def query_image_with_ref(image_path, question, deployment_name="gpt-4.1-mini", example_image_path=None):
    """
    Simple function to query one image with one question, optionally with an example
    """
    chat_model = AzureChat(deployment_name=deployment_name)
    
    if example_image_path:
        # Include both example and target image
        example_base64 = encode_image_to_base64(example_image_path)
        target_base64 = encode_image_to_base64(image_path)
        
        messages = [HumanMessage(content=[
            {
                "type": "text",
                "text": f"Here's an example image for reference:\n\n{question}"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{example_base64}"
                }
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{target_base64}"
                }
            }
        ])]
    else:
        # Original single image approach
        messages = [HumanMessage(content=question)]
        return chat_model(messages, image_path=image_path).content
    
    response = chat_model(messages)
    return response.content
# Example usage
if __name__ == "__main__":
    # Set your image path and question
    image_path = "page_87_200dpi.png"
    example_image_path="Input/example_crop_img.png"
    resolution="2339x1654"
    question = f"""Given an image with resolution {resolution}, please analyze the TARGET image (second image) and provide coordinates to crop the WWAN CNTR device.

                The FIRST image shows an example of proper device cropping.
                
                Requirements:
                - Cover the name of the device itself
                - Cover all connection lines (red lines) extending from the device until its ends
                - Provide coordinates in the format: top-left (x1, y1), bottom-right (x2, y2)

                Note.
                1. the columns are ordered reversely, column 8 is left most and row F is upper most
                
                
                Expected response format:
                # Based on your coordinates: top-left (x1, y1), bottom-right (x2, y2)
                crop_coordinates = (x1, y1, x2, y2)                
                
                please explain step-by-step"""
    #2. The coordinate startes from the top left ends at the bottom right, that is Column 1 starts near x=0, column 8 near end of the resolution; Row F starts near y=0, row A near end of the resolution.
    #- Stop at endpoint pins (IN, OUT, GROUND, or power sources)
    #
    # Query the image
    result = query_image_with_ref(image_path, question, deployment_name="gpt-4.1-mini", example_image_path=example_image_path)
    print(f"Question: {question}")
    print(f"Answer: {result}")