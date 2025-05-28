from crewai import Task

def create_tasks(coordinator, researcher, summarizer, user_query):
    task1 = Task(
        description=f"User asked: '{user_query}'. If it's a question, ask researcher to answer using the PDF content. If it's a summary request, assign to summarizer.",
        expected_output="Clear and helpful response",
        agent=coordinator
    )

    task2 = Task(
        description="Answer the user's question using PDF content only.",
        expected_output="A concise, factual answer.",
        agent=researcher
    )

    task3 = Task(
        description="Summarize the PDF content into a paragraph.",
        expected_output="Concise summary of the document.",
        agent=summarizer
    )

    return [task1, task2, task3]
