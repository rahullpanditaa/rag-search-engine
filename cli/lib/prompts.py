
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