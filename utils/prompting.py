context_instruction = """
--- Instructions:
1. Split the page content into several chunks in Indonesian language. Do this by going through each sentence, if current sentence has similarities with current chunk, add it to current chunk. If not, start a new chunk.
2. For each chunk, split into smaller chunks called facts based on key information within the chunk in Indonesian language.

Provide the output in the following JSON Format:
{
  "chunks": [
    {
      "chunk_id": 1,
      "chunk": "First chunk of the article...",
      "facts": [
        {
          "fact": "Key information within the first chunk in Indonesian language."
        },
        {
          "fact": "Another key information within the first chunk in Indonesian language."
        }
      ]
    },
    {
      "chunk_id": 2,
      "chunk": "Second chunk of the article...",
      "facts": [
        {
          "fact": "Key information within the second chunk in Indonesian language."
        },
        {
          "fact": "Another key information within the second chunk in Indonesian language."
        }
      ]
    },
    // ... More chunks as needed
  ],
  "main_idea": "Main idea of this page in Indonesian language",
}
"""

summary_instruction = """
--- Instructions:

1. Split the page content into several chunks in Indonesian language. Do this by going through each sentence, if current sentence has similarities with current chunk, add it to current chunk. If not, start a new chunk.

Provide the output in the following JSON Format:
{
  "chunks": [
    {
      "chunk_id": 1,
      "chunk": "First chunk of the article..."
    },
    {
      "chunk_id": 2,
      "chunk": "Second chunk of the article..."
    },
    // ... More chunks as needed
  ],
  "summary": "Combine all chunks into short summary in Indonesian language",
}
"""

summary_instruction_rec = """
--- Instructions:

1. Split the page content into about 80 words chunks each by going through each sentence. If current sentence has similarities with current chunk, put it in the same chunk.

Provide the output in the following JSON Format:
{
  "chunks": [
    {
      "chunk_id": 1,
      "chunk": "First chunk of the article..."
    },
    {
      "chunk_id": 2,
      "chunk": "Second chunk of the article..."
    },
    // ... More chunks as needed
  ],
  "summary": "Combine all chunks into short summary in Indonesian language",
}
"""

def context_prompt(pdf_text):
    prompt = f"""
--- Page content:
{pdf_text}

{context_instruction}
"""
    return prompt

def summary_prompt(pdf_text):
    prompt = f"""
--- Page content:
{pdf_text}

{summary_instruction}
"""
    return prompt

def summary_prompt_rec(pdf_text):
    prompt = f"""
--- Page content:
{pdf_text}

{summary_instruction_rec}
"""
    return prompt

fact_instruction = """
--- Instructions:
1. Extract all facts from current page
2. Output as following JSON format in Bahasa Indonesia
{

  "facts": <{"fact": string, "context":<what is this fact about>}[]>

  "docSummary": "<resummarize the document by combining previous page summary and current page content>"

  "context": <create maximum 50 words description about the document based on docSummary>

}
"""

fragment_instruction = """
--- Instructions:
Split the current page content into small, semantically meaningful fragments. Here are the steps:
1. Read through the page content
2. If there is a change of topic, then mark the previous content + some content after the topic separation as a fragment
3. Continue reading for the new fragment, and add some content before the topic separation as the start of the fragment
4. Continue until the end of the page content and the last fragment is concluded

Ensure that the generated fragments include all words from the current page content.

Provide the output in the following JSON Format:

{
  "fragments": [
    {
      "fragment": <Input all the Fragment content referring to the original content, in Indonesian>,
      "topic": <Identify the main topic of the fragment, in Indonesian>,
    },
    // Additional fragments follow the same structure if needed
  ],
  "docSummary": <Create new summary in Indonesian based on previous pages summary and current page content>,
  "context": <Create max 50 words description about the document in Indonesian based on docSummary>
}

OUTPUT ONLY JSON FORMAT according to the given schema
"""

def fact_prompt(pdf_text, cur_summary):
    prompt = f"""
--- Document Data:
Current document summary: {cur_summary}

Current page content: {pdf_text}

{fact_instruction}
"""
    return prompt

def fragment_prompt(pdf_text, cur_summary):
    prompt = f"""
--- Document:
Current document summary: {cur_summary}

Current page content: {pdf_text}

{fragment_instruction}
"""
    return prompt