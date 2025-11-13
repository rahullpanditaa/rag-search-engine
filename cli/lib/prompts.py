
def re_rank_individual_docs_prompt(query: str, doc_title: str, doc_description: str):
    prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc_title} - {doc_description}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""
    return prompt

def re_rank_batch_prompt(query: str, docs: list[dict]):
    prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{docs}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
"""
    return prompt

def llm_evaluation_prompt(query: str, results: list):
    formatted_results = [
        f"{i+1}. {doc['title']} - {doc['document']}" for i, doc in enumerate(results)
    ]
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""
    return prompt

def rag_response_prompt(query: str, docs: list[dict]):
    formatted_docs = "\n".join([
        f"- {doc['title']}: {doc['document']}" for doc in docs
    ])
    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{formatted_docs}

Provide a comprehensive answer that addresses the query:"""
    return prompt