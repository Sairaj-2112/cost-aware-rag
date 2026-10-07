def create_prompt(question, retrieved_chunks):

    context = ""

    for chunk in retrieved_chunks:

        context += f"""
Page {chunk["page"]}:
{chunk["text"]}

"""

    prompt = f"""
You are a helpful study assistant.

Answer the student's question using ONLY the information
provided in the syllabus context below.

If the answer is not present in the context, say:
"I could not find this information in the syllabus."

Syllabus context:
{context}

Student question:
{question}

Give a simple and clear answer.

At the end, write:
Source: Page X

Replace X with the page number from the context that
contains the information used for the answer.
"""

    return prompt