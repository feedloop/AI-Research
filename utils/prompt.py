import os
import tiktoken

from datetime import datetime


def overlap_ratio(str1, str2):
    len_str1 = len(str1)
    len_str2 = len(str2)

    overlap_length = 0
    for i in range(min(len_str1, len_str2)):
        if str1[i:] == str2[: len_str1 - i]:
            overlap_length = len_str1 - i
            break

    total_length = len_str1 + len_str2 - overlap_length
    overlap_ratio = overlap_length / total_length if total_length > 0 else 0

    return overlap_ratio


def overlap_check(new_list, string):
    for existing_string in new_list:
        if (
            overlap_ratio(existing_string, string) >= 0.5
            or overlap_ratio(string, existing_string) >= 0.5
        ):
            return False
    return True


def count_tokens_tiktoken(string, encoding_name="cl100k_base"):
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_base_knowledge_prompt(prompt, iteration, resource_name, info):
    prompt += f"{iteration}. {resource_name}.{info['filetype']}\n"
    prompt += f"Doc URL: {os.getenv('BASE_URL')}{info['project_id']}/{resource_name}.{info['filetype']}\n"

    return prompt


def get_std_knowledge_prompt(memory):
    iteration = 1
    prompt = "--- KNOWLEDGE:\n"
    for resource_name, info in memory.items():
        prompt = get_base_knowledge_prompt(prompt, iteration, resource_name, info)
        iteration += 1

        prompt += f"Facts:\n"
        for _, facts in info["contexts"].items():
            for fact in facts:
                prompt += f"- {fact}\n"

        prompt += "\n"

    return prompt


def get_no_overlap_knowledge_prompt(memory):
    iteration = 1
    prompt = "--- KNOWLEDGE:\n"
    for resource_name, info in memory.items():
        prompt = get_base_knowledge_prompt(prompt, iteration, resource_name, info)
        iteration += 1

        prompt += f"Facts:\n"
        no_overlap_facts = []
        for _, facts in info["contexts"].items():
            for fact in facts:
                if overlap_check(no_overlap_facts, fact):
                    no_overlap_facts.append(fact)

        for fact in no_overlap_facts:
            prompt += f"- {fact}\n"

        prompt += "\n"

    return prompt


def get_auto_merge_knowledge_prompt(memory):
    iteration = 1
    prompt = "--- KNOWLEDGE:\n"
    for resource_name, info in memory.items():
        prompt = get_base_knowledge_prompt(prompt, iteration, resource_name, info)
        iteration += 1

        prompt += f"Facts:\n"
        for context, facts in info["contexts"].items():
            child_len = sum(len(fact) for fact in facts)
            if child_len < len(context) // 2:
                for fact in facts:
                    prompt += f"- {fact}\n"
            else:
                prompt += f"- {context}\n"

        prompt += "\n"

    return prompt


def get_context_knowledge_prompt(memory):
    iteration = 1
    prompt = "--- KNOWLEDGE:\n"
    for resource_name, info in memory.items():
        prompt = get_base_knowledge_prompt(prompt, iteration, resource_name, info)
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
        elif config == "auto_merge":
            cur_prompt = get_auto_merge_knowledge_prompt(sent_knowledge)
        elif config == "no_overlap":
            cur_prompt = get_no_overlap_knowledge_prompt(sent_knowledge)
        else:
            raise ValueError(
                f"config value must be one from ['context', 'standard', 'auto_merge', 'no_overlap']"
            )

        prompt_len = count_tokens_tiktoken(cur_prompt)
        if prompt_len > token_threshold:
            break

    return cur_prompt


def get_knowledge_from_prompt(prompt):
    return [
        r.split(".pdf\n")[-1]
        for r in prompt.split("--- KNOWLEDGE:\n")[-1].split("\n\n")[:-4]
    ]


def get_chat_prompt(
    question, memory, config="context", memory_max_tokens=500, lang="indonesia"
):
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
"language": "{lang}"

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
