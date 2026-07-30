from groq import Groq

from app.config import GROQ_API_KEY, GROQ_MODEL

client = Groq(api_key=GROQ_API_KEY)


def generate_answer(context: str, question: str):
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,
        top_p=0.1,
        max_tokens=64,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an information extraction engine.\n"
                    "Your ONLY source of truth is the supplied context.\n"
                    "Never use outside knowledge.\n"
                    "Never invent facts.\n"
                    "Return only the direct answer.\n"
                    "If the answer is not explicitly present in the context, "
                    "return exactly:\n"
                    "I could not find this information in the document."
                ),
            },
            {
                "role": "user",
                "content": f"""
CONTEXT
--------
{context}

QUESTION
--------
{question}

Extract ONLY the answer from the context.
Do not explain.
Do not summarize.
Do not add extra information.
""",
            },
        ],
        stop=["\n\n"],
    )

    return response.choices[0].message.content.strip()