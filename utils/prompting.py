def context_prompt(pdf_text, language):
    prompt = f"""
--- Page content:
{pdf_text}

--- Instructions:
1. Split the page content into several chunks in {language} language. Do this by going through each sentence, if current sentence has similarities with current chunk, add it to current chunk. If not, start a new chunk.
2. For each chunk, split into smaller chunks called facts based on key information within the chunk in {language} language.

Provide the output in the following JSON Format:
{{
  "chunks": [
    {{
      "chunk_id": 1,
      "chunk": "First chunk of the article...",
      "facts": [
        {{
          "fact": "Key information within the first chunk in {language} language."
        }},
        {{
          "fact": "Another key information within the first chunk in {language} language."
        }}
      ]
    }},
    {{
      "chunk_id": 2,
      "chunk": "Second chunk of the article...",
      "facts": [
        {{
          "fact": "Key information within the second chunk in {language} language."
        }},
        {{
          "fact": "Another key information within the second chunk in {language} language."
        }}
      ]
    }},
    // ... More chunks as needed
  ],
  "main_idea": "Main idea of this page in {language} language",
}}
"""
    return prompt

def summary_prompt(pdf_text, language):
    prompt = f"""
--- Page content:
{pdf_text}

--- Instructions:
1. Split the page content into individual chunks, with each chunk representing a single paragraph. Do not rewrite any part of the content, just segment it into distinct paragraphs.

Provide the output in the following JSON Format:
{{
  "chunks": [
    {{
      "chunk_id": 1,
      "chunk": <First chunk of the article...>
    }},
    {{
      "chunk_id": 2,
      "chunk": <Second chunk of the article...>
    }},
    // ... More chunks as needed
  ],
  "summary": <Combine all chunks into short summary in {language} language>
}}
"""
    return prompt

def summary_prompt_rec(pdf_text, language):
    prompt = f"""
--- Paragraph text:
{pdf_text}

--- Instructions:
1. Split the paragraph text into 2 chunks with equal length. Do not rewrite any part of the content.

Provide the output in the following JSON Format:
{{
  "chunks": [
    {{
      "chunk_id": 1,
      "chunk": <First chunk of the article...>
    }},
    {{
      "chunk_id": 2,
      "chunk": <Second chunk of the article...>
    }}
  ],
  "summary": <Combine all chunks into short summary in {language} language>,
}}
"""
    return prompt

def summary_prompt_rec1(pdf_text, chunk, language):
    return f"""--- Page content:
{pdf_text}

--- Extracted data:
{chunk}

--- Instruction:
Review both the Page Content and the Extracted Data. Provide a concise explanation in {language} language about the context or relation of extracted data to the page content. Provide the output based on JSON format below:
{{
    "context": <your explanation in {language} language here>
}}
"""

def fact_prompt(pdf_text, cur_summary, language):
    prompt = f"""
--- Document Data:
Current document summary: {cur_summary}

Current page content: {pdf_text}

--- Instructions:
1. Extract all facts from current page
2. Output as following JSON format in {language} language
{{

  "facts": <{{"fact": string, "context":<what is this fact about>}}[]>

  "docSummary": "<resummarize the document by combining previous page summary and current page content>"

  "context": <create maximum 50 words description about the document based on docSummary>

}}
"""
    return prompt

def fragment_prompt(pdf_text, cur_summary, language):
    prompt = f"""
--- Document:
Current document summary: {cur_summary}

Current page content: {pdf_text}

--- Instructions:
Split the current page content into small, semantically meaningful fragments. Here are the steps:
1. Read through the page content
2. If there is a change of topic, then mark the previous content + some content after the topic separation as a fragment
3. Continue reading for the new fragment, and add some content before the topic separation as the start of the fragment
4. Continue until the end of the page content and the last fragment is concluded

Ensure that the generated fragments include all words from the current page content.

Provide the output in the following JSON Format:

{{
  "fragments":<{{"fragment":<Input all the Fragment content referring to the original content, in {language} language>,"topic":<Identify the main topic of the fragment, in {language} language>}}[]>,
  "docSummary":<Create new summary in {language} based on previous pages summary and current page content>,
  "context":<Create max 50 words description about the document in {language} based on docSummary>
}}
"""
    return prompt