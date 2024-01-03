import tiktoken
from datetime import datetime


def count_tokens_tiktoken(string, encoding_name="cl100k_base"):
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_std_knowledge_prompt(memory):
    iteration = 1
    prompt = "--- KNOWLEDGE:\n"
    for resource_name, info in memory.items():
        prompt += f"{iteration}. {resource_name}.{info['filetype']}\n"
        prompt += f"Doc URL: https://platform-staging-api.feedloop.ai/download/{info['project_id']}/{resource_name}.{info['filetype']}\n"
        iteration += 1

        for _, facts in info["contexts"].items():
            prompt += f"Facts:\n"
            for fact in facts:
                prompt += f"- {fact}\n"

        prompt += "\n"

    return prompt


def get_context_knowledge_prompt(memory):
    iteration = 1
    prompt = "--- KNOWLEDGE:\n"
    for resource_name, info in memory.items():
        prompt += f"{iteration}. {resource_name}.{info['filetype']}\n"
        prompt += f"Doc URL: https://platform-staging-api.feedloop.ai/download/{info['project_id']}/{resource_name}.{info['filetype']}\n"
        iteration += 1

        count_ctx = 1
        for context, facts in info["contexts"].items():
            prompt += f"Context {count_ctx}: {context}\n"
            prompt += f"Facts from Context {count_ctx}:\n"
            for fact in facts:
                prompt += f"- {fact}\n"
            count_ctx += 1

        prompt += "\n"

    return prompt


def get_knowledge_prompt(memory, config="context", token_threshold=500):
    sent_knowledge = {}
    cur_prompt = ""
    for r in memory:
        if r["name"] not in sent_knowledge.keys():
            sent_knowledge[r["name"]] = {
                "contexts": {r["context"]: []},
                "project_id": r["project"],
                "filetype": r["filetype"],
            }
        if r["context"] not in sent_knowledge[r["name"]]["contexts"].keys():
            sent_knowledge[r["name"]]["contexts"][r["context"]] = []

        sent_knowledge[r["name"]]["contexts"][r["context"]].append(r["fact"])

        if config == "context":
            cur_prompt = get_context_knowledge_prompt(sent_knowledge)
        elif config == "standard":
            cur_prompt = get_std_knowledge_prompt(sent_knowledge)

        prompt_len = count_tokens_tiktoken(cur_prompt)
        if prompt_len > token_threshold:
            break

    return cur_prompt


def get_chat_prompt(question, memory, config="context", memory_max_tokens=500):
    return f"""[BOT]
--- SYSTEM INFO:
Today's Time: {datetime.now().strftime("%A, %d %B %Y %I:%M%p")}
--- BOT INSTRUCTION

--- USER INFO:
{{"name":"John","email":"john@feedloop.ai"}}

{get_knowledge_prompt(memory, config=config, token_threshold=memory_max_tokens)}--- CONTEXT SCHEMA:
type context ={{
}}
--- PRIOR USER CONTEXT:
{{}}
--- CONVERSATION STYLE:
"language": "indonesia"

--- PRIOR CONVERSATION:
prior conversation summary: 
user: 
assistance:
--- USER MESSAGE:
{question}

---
ANSWER PROPERLY WITH FOLLOWING JSON FORMAT!:
{{
  "response": {{text: <RESPONSE THE USER MESSAGE HERE USING THE LANGUAGE OF THE USER, Please generate a response that is easy to read by adding a newline (\n\n) after every sentence>, ref: <IF THE RESPONSE USING KNOWLEDGE, write <knowledge no>, if no then null>}}[] <Array of object>,
  "references": {{
      "<knowledge no>": {{
      "name": <document name>,
      "url": <document url>
    }}
  }} -> only fill reference if your response based on KNOWLEDGE section otherwise empty!,
  "contextOps": <array of JSON Patch operations (RFC 6902): based on user message and previous conversation and properties types, please update context object and consider following properties to update: none. DO NOT UPDATE ANY PROPERTIES NOT IN THE SCHEMA!>,
  "replySuggestions": <given bot response, suggest max 3 possible reply for user to choose from, type: string[]>
}}
ANSWER ONLY AS JSON WITH ABOVE DEFINED FORMAT!"""
