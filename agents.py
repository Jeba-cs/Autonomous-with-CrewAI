# agents.py

from crewai import Agent
from crewai.tools import tool
from openai import OpenAI
from pdf_utils import search_chunks

client = OpenAI(api_key="", base_url="https://api.deepseek.com")

@tool
def answer_question(query: str, chunks: list, index) -> str:
    """Answer a user's question using relevant content extracted from the uploaded PDF."""
    context = "\n".join(search_chunks(query, index, chunks))
    prompt = f"Use the following PDF content to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        stream=False,
    )
    return response.choices[0].message.content.strip()

@tool
def summarize_pdf(_: str, chunks: list, index) -> str:
    """Summarize the entire content of the uploaded PDF in a clear and concise way."""
    content = "\n".join(chunks)
    prompt = f"Summarize the following document:\n\n{content}"
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        stream=False,
    )
    return response.choices[0].message.content.strip()

def create_agents(chunks, index):
    researcher = Agent(
        role="PDF Researcher",
        goal="Answer questions accurately using the PDF data.",
        backstory="Expert in document analysis and research.",
        tools=[answer_question],
        verbose=True
    )

    summarizer = Agent(
        role="PDF Summarizer",
        goal="Summarize the content of the PDF clearly.",
        backstory="Expert in summarizing documents.",
        tools=[summarize_pdf],
        verbose=True
    )

    coordinator = Agent(
        role="Coordinator",
        goal="Decide if summarization or question answering is needed based on user query.",
        backstory="Intelligent agent responsible for delegating tasks.",
        verbose=True
    )

    return coordinator, researcher, summarizer
