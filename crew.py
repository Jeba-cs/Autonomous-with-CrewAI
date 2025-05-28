# crew.py

from crewai import Crew, Task
from agents import create_agents

def run_crew(chunks, index, user_query):
    coordinator, researcher, summarizer = create_agents(chunks, index)

    research_task = Task(
        description="Answer the question from the user using the PDF content: " + user_query,
        expected_output="A precise and helpful answer using the document content.",
        agent=researcher
    )

    summary_task = Task(
        description="Summarize the entire PDF content.",
        expected_output="A concise and informative summary of the PDF.",
        agent=summarizer
    )

    # Coordinator task (to let CrewAI decide automatically, if you're using one)
    # Optional: can be created from `tasks.py` or add here
    coordinator_task = Task(
        description="Decide whether the user query needs summarization or question answering, then delegate the task.",
        expected_output="A decision on which path to take, and a completed response from the chosen path.",
        agent=coordinator
    )

    crew = Crew(
        agents=[coordinator, researcher, summarizer],
        tasks=[coordinator_task, research_task, summary_task],
        verbose=True
    )

    result = crew.kickoff(inputs={"query": user_query})
    return result
