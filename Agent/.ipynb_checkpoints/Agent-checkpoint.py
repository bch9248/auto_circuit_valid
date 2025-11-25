import requests            # For making HTTP POST requests to Azure/OpenAI or Local API
import time                # For sleep (retry delays)
import os
from langchain.schema import BaseMessage, AIMessage  # LangChain message schema
import requests
import time
from langchain.schema import BaseMessage, HumanMessage, AIMessage

def convert_langchain_messages(messages):
    role_map = {
        "system": "system",
        "human": "user",
        "ai": "assistant"
    }
    return [{"role": role_map[msg.type], "content": msg.content} for msg in messages]

def is_filtered_by_azure(response_json):
    """
    Returns True if Azure filtered the message (regardless of severity).
    """
    return response_json["choices"][0].get("finish_reason") == "content_filter"


class AzureChat:
    def __init__(self, deployment_name, temperature=0.7, max_tokens=1000, api_version="2025-01-01-preview"):
        if deployment_name == "gpt-35-turbo":
            self.api_url = f"https://gpt4fordg.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2025-01-01-preview"
            api_key=os.getenv("OPENAI_API_KEY_GPT35")
        elif deployment_name == "gpt-4o-mini":
            self.api_url = f"https://for-dc-test.openai.azure.com/openai/deployments/{deployment_name}/chat/completions?api-version=2025-01-01-preview"
            api_key = os.getenv("OPENAI_API_KEY_GPT4o")
        elif deployment_name == "gpt-4.1-mini":
            self.api_url = f"https://aoai-sc-aic-016.openai.azure.com/openai/deployments/gpt-4.1-mini/chat/completions?api-version=2025-01-01-preview"
            api_key = os.getenv("OPENAI_API_KEY_GPT41")
        elif deployment_name == "gpt-4.1":
            self.api_url = f"https://aoai-sc-dc-011.openai.azure.com/openai/deployments/gpt-4.1/chat/completions?api-version=2025-01-01-preview"
            api_key = os.getenv("OPENAI_API_KEY_GPT41")
        else:                    
            self.api_url = f"https://for-dc-test.openai.azure.com/openai/deployments/{deployment_name}/chat/completions?api-version=2024-08-01-preview"
            api_key = os.getenv("OPENAI_API_KEY_GPT4o")
        self.headers = {
            "Content-Type": "application/json",
            "api-key": api_key
        }
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(self, messages):
        role_map = {
            "system": "system",
            "human": "user",
            "ai": "assistant"
        }

        formatted_messages = []
        for msg in messages:
            if isinstance(msg, BaseMessage):
                formatted_messages.append({
                    "role": role_map.get(msg.type, "user"),
                    "content": msg.content
                })
            elif isinstance(msg, dict):
                formatted_messages.append(msg)
            else:
                raise ValueError(f"Unsupported message format: {msg}")

        payload = {
            "messages": formatted_messages,
            "temperature": self.temperature,
            #"max_tokens": self.max_tokens
        }
        max_retries = 5
        delay = 5
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload)
                response.raise_for_status()        
#                 print(response.json())
                response_json = response.json()
#                 print(response)
                if is_filtered_by_azure(response_json):  # handle when azure api reject 
                    print(f"message_filtered by azure due to safety issue")
                    ignore_msg="unsafety_input"
                    return AIMessage(content=ignore_msg)  # ✅ wrap in LangChain message
                else:
                    output = response.json()["choices"][0]["message"]["content"]
                    return AIMessage(content=output)  # ✅ wrap in LangChain message
            except Exception as e:
                print(f"[Attempt {attempt+1}] Error: {e}")
#                 response = requests.post(self.api_url, headers=self.headers, json=payload)
#                 results = response["choices"][0]["content_filter_results"]
#                 for category, data in results.items():
#                     if data["severity"] == "high":
#                         return True  # Only block on high
                
                time.sleep(delay)
                delay *= 2  # Exponential backoff

        raise RuntimeError("Azure OpenAI API call failed after retries")

class LocalChat:
    def __init__(self, host="http://10.15.10.22:11434/api/generate", model="deepseek-r1:7b"):
        self.host = host
        self.model = model

    def __call__(self, messages):
        # Handle LangChain messages
        prompt = "\n".join([
            f"{self._role_name(m)}: {m.content}" for m in messages if isinstance(m, BaseMessage)
        ])

        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        max_retries = 5
        delay = 2
        for attempt in range(max_retries):
            response = requests.post(self.host, json=data)

            if response.status_code == 200:
                output = response.json()["response"]
                return AIMessage(content=output)  # ⬅️ fix here
            elif response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    delay = int(retry_after)
                print(f"[Retry {attempt+1}] Rate limit. Waiting {delay}s...")
                time.sleep(delay)
            else:
                print(f"[Retry {attempt+1}] Error {response.status_code}: {response.text}")
                break

        raise RuntimeError("Local LLM request failed after retries")

    def _role_name(self, msg):
        if msg.type == "system":
            return "System"
        elif msg.type == "human":
            return "User"
        elif msg.type == "ai":
            return "Assistant"
        return "Unknown"


sents_array="""0. President Trump said Friday that he will support a short-term spending bill to re-open the government, temporarily ending the partial government shutdown that has dragged on for more than a month -- despite a day earlier saying Republicans would not “cave” on his demands for funding for a wall on the southern border.
1. “I am very proud to announce we have reached a deal to end the shutdown and reopen the federal government,” he said to applause from gathered Cabinet members.
2. He said the deal would keep the government open for three weeks until Feb. 15 and that a bill would go before the Senate immediately. 
3. He made reference to his previous threats to declare a national emergency, calling it a “very powerful weapon,” but saying he didn’t want to use it. 
4. The deal appeared to include no money for a wall or steel barrier, but he said he hoped negotiations would continue to come to an agreement on wall funding.
5. "Walls should not be controversial," he said.
6. The shutdown was sparked by disagreement over President Trump’s demand for $5.7 billion in funding for a wall or steel barrier on the southern border. 
7. Democrats countered initially with $1.3 billion for general border security, while Trump said initially he would not sign anything without wall funding.
8. In the last few weeks, the fight turned particularly nasty -- with Trump canceling a Democratic congressional trip to Afghanistan after House Speaker Nancy Pelosi called on Trump to delay his State of the Union address or submit it in writing. 
9. Trump announced Thursday that he would delay the address, but if the government re-opens then it could take place on Tuesday as had been previously scheduled.
10. After bills to re-open the government failed in the Senate Thursday, Trump signaled that a "large down payment" on funding, potentially less than the $5.7 billion, could be sufficient to end the stalemate. 
11. He suggested that a "prorated down payment" on the wall, without providing a concrete dollar figure, could be viable.
12. Sen. Lindsey Graham, R-S.C., had also suggested that a three-week continuing resolution could be the way forward.
13. "If they come to a reasonable agreement, I would support that," Trump told reporters.
14. The partial closure of the government has led to hundreds of thousands of workers furloughed or working without pay, with groups representing the workers increasing their calls for D.C. to end the deadlock and get workers paid again. 
15. Many workers missed their second paycheck on Friday.
16. Every former secretary of the Department of Homeland Security, including former White House chief of staff John Kelly, sent a letter to the president and Congress on Thursday asking them restore the department’s funding.
17. “DHS employees who protect the traveling public, investigate and counter terrorism, and protect critical infrastructure should not have to rely on the charitable generosity of others for assistance in feeding their families and paying their bills while they steadfastly focus on the mission at hand,” the letter said. 
18. “This is unconscionable.”
19. President Trump on Saturday announced a compromise plan that funded the wall while extending protections for 700,000 illegal immigrants brought to the country as children and 300,000 immigrants from countries designated unsafe to return. 
20. However, Democrats rejected the plan even before Trump announced it, and a Senate version of the plan failed to get the 60 votes needed on Thursday. 
21. A second bill, already passed by the Democrat-controlled House to re-open the government, also fell short.
22. On Thursday, Trump had promised that Republicans “will not cave" on border wall demands.
  ← (presented as a numbered list: 0. ..., 1. ..., 2. ...)"""    
    
if __name__ == "__main__":
#     The below content was filtered by azure openai due to the error message: 
    

#     eval_prompt="""You are a media analyst detecting bias in news writing. A biased sentence with political bias refers to content—whether supportive, critical, or suggestive—that sways readers' opinions toward an ideological direction.

# ---

# For each sentence:

# * Briefly explain why you think it's biased.
# * Decide whether the sentence is biased (Yes/No).
# * Provide your confidence in your decision as a numerical percentage (0-100%). Use uncertainty adjectives (e.g., "possibly," "likely," "probably," "definitely") explicitly to reflect your confidence:

#   * 0-30%: "possibly biased/unbiased"
#   * 31-60%: "likely biased/unbiased"
#   * 61-85%: "probably biased/unbiased"
#   * 86-100%: "definitely biased/unbiased"

# At the end, output only the list of indices of biased sentences along with their confidence scores in this format:
# **\[\[index1: confidence1%, index2: confidence2%, index3: confidence3%]]**

# ---

# **Sentences to evaluate:**

# {sents_array}

# ---


# **Answer (analyze step by step, then finalize list):**

# {sents\_array}  ← (presented as a numbered list: 0. ..., 1. ..., 2. ...)

# ---

# **Answer:**
# """.format(sents_array=sents_array)
    
    eval_prompt = """You are a media analyst detecting bias in news writing. A biased sentence with political bias refers to content—whether supportive, critical, or suggestive—that sways readers' opinions toward an ideological direction.

---

Instructions:
For each sentence:
1. Copy the full sentence exactly as shown.
2. Repeat the sentence index and sentence text in the output.
3. Classify the sentence as biased: "Yes" or "No"
4. Assign a confidence score (0–100) reflecting your certainty.
5. Start your explanation with a **hedging phrase** (choose one from this exact list):

"almost certain", "highly likely", "very likely", "probable", "somewhat likely", "possible",  
"uncertain", "somewhat unlikely", "unlikely", "not likely", "doubtful", "very unlikely", "highly unlikely"

Each output block **must strictly follow this format**:

Sentence {{i}}: {{original sentence}}  
Bias Decision: [Yes or No]  
Confidence: [0–100]%  
Explanation: [Start with one hedging word, then brief justification]

Repeat this block for each sentence, with no missing fields and no extra commentary.

---

At the end, print the list of indices classified as biased using this exact format:
**[[index1, index2, index3]]**

---

**Sentences to evaluate:**

{sents_array}  ← (format: 0. ..., 1. ..., 2. ...)

---

**Answer :**
""".format(sents_array=sents_array)


    # --- AzureChat setup ---
    # deployment_name = "gpt-4.1-mini"  # or your own deployment
    deployment_name = "gpt-35-turbo"  # or your own deployment

    chat_model = AzureChat(deployment_name=deployment_name)

    # --- Construct message list ---
    #eval_prompt
    messages = [HumanMessage(content=eval_prompt)]

    # --- Call the model ---
    response = chat_model(messages)

    # --- Print the output ---
    print("=== AzureChat Response ===")
    print(response.content)

    print(response)    
