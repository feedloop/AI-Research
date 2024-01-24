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


def get_base_facts_prompt(prompt, iteration, resource_name, info):
    prompt += f"{iteration}. {resource_name}.{info['filetype']}\n"
    prompt += f"Doc URL: {os.getenv('BASE_URL')}{info['project_id']}/{resource_name}.{info['filetype']}\n"

    resource_type = resource_name.split("- ")[-1]
    return prompt, resource_type


def get_std_facts_prompt(info):
    prompt += f"Facts:\n"
    for _, facts in info["contexts"].items():
        for fact in facts:
            prompt += f"- {fact}\n"

    return prompt


def get_no_overlap_facts_prompt(info):
    prompt = f"Facts:\n"
    no_overlap_facts = []
    for _, facts in info["contexts"].items():
        for fact in facts:
            if overlap_check(no_overlap_facts, fact):
                no_overlap_facts.append(fact)

    for fact in no_overlap_facts:
        prompt += f"- {fact}\n"

    return prompt


def get_auto_merge_facts_prompt(info):
    prompt = f"Facts:\n"
    for context, facts in info["contexts"].items():
        child_len = sum(len(fact) for fact in facts)
        if child_len < len(context) // 2:
            for fact in facts:
                prompt += f"- {fact}\n"
        else:
            prompt += f"- {context}\n"

    prompt += "\n"

    return prompt


def get_context_facts_prompt(info):
    prompt = ""
    count_ctx = 1
    for context, facts in info["contexts"].items():
        prompt += f"Context {count_ctx}: {context}\n"
        prompt += f"Facts from Context {count_ctx}:\n"
        for fact in facts:
            prompt += f"- {fact}\n"
        count_ctx += 1

    prompt += "\n"

    return prompt


def get_facts_prompt(memory):
    iteration = 1
    prompt = "--- KNOWLEDGE:\n"
    for resource_name, info in memory.items():
        prompt, resource_type = get_base_facts_prompt(
            prompt, iteration, resource_name, info
        )
        iteration += 1

        if "ParentChild" in resource_type:
            facts_prompt = get_auto_merge_facts_prompt(info)
        elif "SentWindow" in resource_type:
            facts_prompt = get_no_overlap_facts_prompt(info)
        elif "context_fact" in resource_type or "summary_fact" in resource_type:
            facts_prompt = get_context_facts_prompt(info)
        else:
            facts_prompt = get_std_facts_prompt(info)

        prompt += facts_prompt
        prompt += "\n"

    return prompt


def get_knowledge_prompt(memory, token_threshold=500):
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

        cur_prompt = get_facts_prompt(sent_knowledge)
        prompt_len = count_tokens_tiktoken(cur_prompt)
        if prompt_len > token_threshold:
            break

    return cur_prompt


def get_knowledge_from_prompt(prompt):
    return [
        r.split(".pdf\n")[-1]
        for r in prompt.split("--- KNOWLEDGE:\n")[-1].split("\n\n")[:-4]
    ]


def get_chat_prompt(question, memory, memory_max_tokens=500, lang="indonesia"):
    return f"""[BOT]
--- SYSTEM INFO:
Today's Time: {datetime.now().strftime("%A, %d %B %Y %I:%M%p")}
--- BOT INSTRUCTION

--- USER INFO:
{{"name":"John","email":"john@feedloop.ai"}}

{get_knowledge_prompt(memory, token_threshold=memory_max_tokens)}--- CONTEXT SCHEMA:
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

def get_eval_prompt(question, expected_answer, answer):
    data = f"""Question:
{question}
---

Expected answer:
{expected_answer}
---

Current answer:
{answer}
---
"""
    json_format = """
Evaluate Current answer compared to Expected answer. Response with following valid JSON format:

{{ 
    "similarity": <similarity percentage of current answer to expected answer>,

    "correctness": <correctness percentage of current answer to expected answer>,

    "relevancy: <relevancy percentage of current answer to question>,

    "reasons": <explain reasons for the evaluation in about 50 words>

}}
"""
    return data+json_format

def get_eval_ragas_prompt(question, ground_truth, answer, context):
    return f"""--- Question:
{question}

--- Ground Truth:
{ground_truth}

--- Answer:
{answer}

--- Context:
{context}

--- Task 1 Rate Precision score:
Analyze the question, answer, and context then determine whether the context provides essential information that help in answering the question. If yes then Precision score is 1, else 0.

--- Task 2 Calculate Recall score by following steps below:
1. Identify True Positives (TP) value by counting the number of sentences in the answer that correctly align with and can be attributed to the context.
2. Divide the TP value (from step 1) by the total number of sentences in the answer. The result is the Recall score.

--- Task 3 Calculate F1 score by following steps below:
1. Identify TP (the number of statements present in both the answer and the ground truth).
2. Identify FP (the number of statements in the answer but not found in the ground truth).
3. Identify FN (the number of statements in the ground truth but not found in the answer).
4. Calculate F1 score = TP / (TP + 0.5 * (TP + FN))

--- Output:
Output task 1, 2, and 3 result in below JSON format:
{{
    "precision": <Precision score result>,

    "recall": <Recall score result>,

    "f1_score": <F1 score result>
}}
"""

def get_eval_ragas_pr(question, answer, context):
    return f"""--- Question:
{question}

--- Answer:
{answer}

--- Context:
{context}

--- Task 1 Rate Precision score:
Analyze the question, answer, and context then verify if the context was useful in arriving at the given answer. Give Precision score as 1 if useful, else 0.

--- Task 2 Calculate Recall score by following steps below:
1. Identify True Positives (TP) value by counting the number of sentences in the answer that can be attributed to the given context.
2. Divide the TP value (from step 1) by the total number of sentences in the answer. The result is the Recall score.

--- Output:
Output task 1 and 2 result in below JSON format:
{{
    "precision": <Precision score result>,

    "recall": <Recall score result>
}}
"""

def get_eval_ragas_cr(question, ground_truth, answer):
    return f"""--- Question:
{question}

--- Ground Truth:
{ground_truth}

--- Answer:
{answer}

--- Task 1 Rate Relevancy score by following steps below:
1. Generate a question based on the given answer.
2. Rate relevancy score by comparing relevancy of the generated question to given question. Score can range from 0 to 1 with 1 being the best.

--- Task 2 Calculate F1 score by following steps below:
1. Identify TP (the number of statements that are present in both the answer and the ground truth).
2. Identify FP (the number of statements present in the answer but not found in the ground truth).
3. Identify FN (the number of statements found in the ground truth but omitted in the answer).
4. Calculate F1 score = TP / (TP + 0.5 * (TP + FN))

--- Output:
Output task 1 and 2 result in below JSON format:
{{
    "relevancy": <Relevancy score result>,

    "f1_score": <F1 score result>
}}
"""

def get_eval_ragas_pr1(question, answer, context):
    return f"""--- Question:
{question}

--- Answer:
{answer}

--- Context:
{context}

--- Task 1 Rate Precision score:
Analyze the question, answer, and context then determine whether the context provides essential information that help in answering the question. If yes then Precision score is 1, else 0.

--- Task 2 Calculate Recall score by following steps below:
1. Identify True Positives (TP) value by counting the number of sentences in the answer that correctly align with and can be attributed to the context.
2. Divide the TP value (from step 1) by the total number of sentences in the answer. The result is the Recall score.

--- Output:
Output task 1 and 2 result in below JSON format:
{{
    "precision": <Precision score result>,

    "recall": <Recall score result>
}}
"""

def get_eval_ragas_cr1(question, ground_truth, answer):
    return f"""--- Question:
{question}

--- Ground Truth:
{ground_truth}

--- Answer:
{answer}

--- Task 1 Rate Relevancy score by following steps below:
1. Generate a question based on the given answer.
2. Rate relevancy score by comparing relevancy of the generated question to given question. Score can range from 0 to 1 with 1 being the best.

--- Task 2 Calculate F1 score by following steps below:
1. Identify TP (the number of statements present in both the answer and the ground truth).
2. Identify FP (the number of statements in the answer but not found in the ground truth).
3. Identify FN (the number of statements in the ground truth but not found in the answer).
4. Calculate F1 score = TP / (TP + 0.5 * (TP + FN))

--- Output:
Output task 1 and 2 result in below JSON format:
{{
    "relevancy": <Relevancy score result>,

    "f1_score": <F1 score result>
}}
"""

def get_eval_ragas_pr2(question, answer, context):
    return f"""--- Question:
{question}

--- Answer:
{answer}

--- Context:
{context}

--- Instruction:
Output in below JSON format:
{{
    "precision": <Analyze the question, answer, and context then determine whether the context provides essential information that help in answering the question. If yes then Precision score is 1, else 0.>,

    "tp": <Count the number of sentences in the answer that correctly align with and can be attributed to the context.>,

    "total": <Total sentences in the answer.>
}}
"""

def get_eval_ragas_cr2(question, ground_truth, answer):
    return f"""--- Question:
{question}

--- Ground Truth:
{ground_truth}

--- Answer:
{answer}

--- Instruction:
Output in below JSON format:
{{
    "generated_question": <Generate a question based on the given answer.>,

    "relevancy": <Rate the relevancy of the generated question to the given question, score can range from 0 to 1 with 1 being the best.>,

    "tp": <Count the number of statements present in both the answer and the ground truth.>,

    "fp": <Count the number of statements in the answer but not found in the ground truth.>,

    "fn": <Count the number of statements in the ground truth but not found in the answer.>
}}
"""

def get_eval_ragas_pr_detail1(question, answer, context):
    input = f"""--- Question:
{question}

--- Answer:
{answer}

--- Context:
{context}
"""
    
    task = """
--- Task 1:
From the given context and answer, analyze each sentence in the answer and classify if the sentence can be attributed to the given context or not. Use only Yes (1) or No (0) as a binary classification. Output json with reason. Refer to example1 below on how to perform the task

example1={
    "question": "What can you tell me about albert Albert Einstein?",
    "context": "Albert Einstein (14 March 1879 - 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics 'for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect', a pivotal step in the development of quantum theory.",
    "answer": "Albert Einstein born in 14 March 1879 was  German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his services to theoretical physics. He published 4 papers in 1905.  Einstein moved to Switzerland in 1895",
    "classification": [
        {
            "statement_1": "Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time.",
            "reason": "The date of birth of Einstein is mentioned clearly in the context.",
            "Attributed": 1,
        },
        {
            "statement_2": "He received the 1921 Nobel Prize in Physics 'for his services to theoretical physics.",
            "reason": "The exact sentence is present in the given context.",
            "Attributed": 1,
        },
        {
            "statement_3": "He published 4 papers in 1905.",
            "reason": "There is no mention about papers he wrote in the given context.",
            "Attributed": 0,
        },
        {
            "statement_4": "Einstein moved to Switzerland in 1895.",
            "reason": "There is no supporting evidence for this in the given context.",
            "Attributed": 0,
        },
    ]
}

--- Task 2:
From the given question, answer and context verify if the context was useful in arriving at the given answer. Give verdict as 1 if useful, else 0. Output json with reason. Refer to example2 below on how to perform the task

example2=[
    {
        "question": "who won 2020 icc world cup?",
        "context": "Who won the 2022 ICC Men's T20 World Cup?",
        "answer": "England",
        "verification": {
            "reason": "the context was useful in clarifying the situation regarding the 2020 ICC World Cup and indicating that England was the winner of the tournament that was intended to be held in 2020 but actually took place in 2022.",
            "verdict": 1,
        },
    },
    {
        "question": "What is the tallest mountain in the world?",
        "context": "The Andes is the longest continental mountain range in the world, located in South America. It stretches across seven countries and features many of the highest peaks in the Western Hemisphere. The range is known for its diverse ecosystems, including the high-altitude Andean Plateau and the Amazon rainforest.",
        "answer": "Mount Everest.",
        "verification": {
            "reason": "the provided context discusses the Andes mountain range, which, while impressive, does not include Mount Everest or directly relate to the question about the world's tallest mountain.",
            "verdict": 0,
        },
    },
]

--- Output:
Combine task 1 and 2 json output into single json with structure like below:
{
    "classification": <task 1 json>[],

    "verification": {"reason":<verdict reason from task 2>,"verdict":<verdict score from task 2>}
}
"""
    return input+task

def get_eval_ragas_pr_detail2(question, answer, context):
    input = f"""--- Question:
{question}

--- Answer:
{answer}

--- Context:
{context}
"""
    
    task = """
--- Task 1:
From the given context and answer, analyze each sentence in the answer and classify if the sentence can be attributed to the given context or not. Use only Yes (1) or No (0) as a binary classification. Refer to JSON_example1 below for output.

JSON_example1 = {"classification": {"statement": <sentence from answer>, "reason": <reason for classification>, "Attributed": <classification score>}[]}

--- Task 2:
From the given question, answer and context verify if the context was useful in arriving at the given answer. Give verdict as 1 if useful, else 0. Refer to JSON_example2 below for output.

JSON_example2= {"verification": {"reason": <reason for verdict>, "verdict": <verdict score>}

--- Output:
Combine task 1 and 2 JSON output into single json with structure like below:
{
    "classification": <task 1 JSON output>[],

    "verification": <task 2 JSON output>
}
"""
    return input+task

def get_eval_ragas_pr_detail3(question, answer, context):
    input = f"""--- Question:
{question}

--- Answer:
{answer}

--- Context:
{context}
"""
    
    task = """
--- Task 1 (Classification):
Analyze each sentence in the provided answer and classify whether the sentence can be attributed to the given context. Use a binary classification system with only "Yes (1)" or "No (0)". Each sentence in the answer should be treated as a separate entry in the JSON array. In cases of ambiguity, classify as "No (0)" and provide a reason for this decision. Provide output based on JSON_example1 below

JSON_example1 = {
    "classification": [
        {
            "statement_1": "<First sentence from answer>",
            "reason": "<Explanation for classification>",
            "Attributed": "<1 or 0>"
        },
        {
            "statement_2": "<Second sentence from answer>",
            "reason": "<Explanation for classification>",
            "Attributed": "<1 or 0>"
        }
        // ... additional sentences
    ]
}

--- Task 2 (Verification):
From the given question, answer and context verify if the context was useful in arriving at the given answer. Give verdict as "1" if useful and "0" if not. Partially useful should be classified as "1". Provide output based on JSON_example2 below

JSON_example2 = {
    "verification": {
        "reason": "<Explanation for verdict>",
        "verdict": "<1 or 0>"
    }
}

--- Output:
Combine the JSON outputs from Task 1 (Classification) and Task 2 (Verification) into a single JSON object. The final JSON should have two keys: "classification" for the array of classifications from Task 1, and "verification" for the verification result from Task 2. Refer to JSON structure below

{
    "classification": [
        // Array of classification objects from Task 1
        // Example: {"statement": "Example sentence", "reason": "Example reason", "Attributed": 1}
    ],
    "verification": {
        // Verification object from Task 2
        // Example: {"reason": "Example reason for verdict", "verdict": 1}
    }
}
"""
    return input+task

def get_eval_ragas_cr_detail1(question, ground_truth, answer): 
    return f"""--- Question:
{question}

--- Ground Truth:
{ground_truth}

--- Answer:
{answer}

--- Instruction:
Provide output based on JSON structure below:
{{
    "relevancy": "Identify if the given answer is noncommittal to the given question. Rate 1 if noncommittal, else 0.",

    "gen_question": "Generate a question in the answer language that is directly related to the given answer.",

    "tp": "List statements that are present in both the answer and the ground truth. These are true positive statements, indicating accurate and relevant information found in both sources.",

    "fp": "List statements that are present in the answer but not found in the ground truth. These are false positive statements, indicating information in the answer that is not corroborated by the ground truth.",

    "fn": "List statements that are present in the ground truth but not found in the answer. These are false negative statements, indicating relevant information omitted in the answer."
}}
"""

def get_eval_ragas_cr_trial1(question, answer):
    return f"""--- Question:
{question}

--- Answer:
{answer}

--- Instruction:
Identify if the given answer is noncommittal to the given question. Rate 1 if noncommittal, else 0. Provide output based on JSON structure below:
{{
    "relevancy": <1 or 0>
}}
"""

def style_transfer2(prompt):
    return f"""--- Instruction:
{prompt} in Indonesian language. Provide output based on JSON structure below:
{{
    "formal": <generated paragraph>

    "informal": <rewrite each sentence from the generated paragraph in Indonesian language, informal style. Total sentences must be the same.>

    "alay": <rewrite each sentence from the generated paragraph in Indonesian language, alay style. Total sentences must be the same.>

    "jaksel": <rewrite each sentence from the generated paragraph in Indonesian language, jaksel style. Total sentences must be the same.>
}}"""

# Generate a 5-sentences paragraph about 

def style_transfer3(prompt):
    return f"""--- Instruction:
{prompt} in Indonesian language. Provide output based on JSON structure below:
{{
    "formal": <generated questions>

    "informal": <rewrite each question from the generated questions in Indonesian language, informal style. Total questions must be the same.>

    "alay": <rewrite each question from the generated questions in Indonesian language, alay style. Total questions must be the same.>

    "jaksel": <rewrite each question from the generated questions in Indonesian language, jaksel style. Total questions must be the same.>
}}
"""

def style_transfer4(prompt):
    return f"""--- Instruction:
{prompt} in Indonesian language. Provide output based on JSON structure below:
{{
    "formal": <generated text conversation>

    "informal": <rewrite each sentence from the generated text in Indonesian language, informal style. Total sentences must be the same.>

    "alay": <rewrite each sentence from the generated text in Indonesian language, alay style. Total sentences must be the same.>

    "jaksel": <rewrite each sentence from the generated text in Indonesian language, jaksel style. Total sentences must be the same.>
}}"""