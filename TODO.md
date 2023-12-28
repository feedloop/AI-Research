# TODO

- Add all doc with desired stored types into FL platform (new fact, langchain/llama_index default text splitter, fragment ext, summary)
- Run each stored type in python, add it to fact table with its corresponding pdf resource_id

## New Fact Extraction Prompt Test

According to prev. discussion. Can be added by using Ragas eval

## RAG Test

Generate question eval query with its expected answer criteria for every

Alternative: Generate test set via Ragas TestSetGenerator

- Test with FL Platform
  - Utilize the expected answer criteria of query
  - Run agent with same configuration on different resource stored types
  - Calculate overall test results and compare
- Test using Ragas Framework
  - Generation metric
    - Run the question eval query with same RAG pipeline but different stored types
    - Evaluate the LLM response based on query and given context
    - Measure faithfulness and answer relevance from the response
    - Using GPT-4 as evaluator
  - Retrieval Metric
    - Run the question eval query to retrieve context
    - Evaluate the retrieved context
    - Measure Precision (How many contexts are relevant with query?) and Recall (How much the retrieved contexts covered the query?)
    - Witth Precision and Recall can calculate F1-Score
    - Using GPT-4 as evaluator
