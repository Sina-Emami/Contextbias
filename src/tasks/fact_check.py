from crewai import Task

FACT_CHECK_PROMPT = (
    "Search online using your tools and answer the question with a SINGLE percentage (0–100%), "
    "a single one-word category, or YES/NO, plus a citation (URL or publication). Prefer reputable/official stats.\n\n"
    "If no reliable/recent source is found, return NOT FOUND.\n\n"
    "Return your FULL output as a JSON LIST with ONE object exactly like:\n"
    '[{"question":"<original>","answer":"<percentage | one-word | YES/NO | NOT FOUND>","source":"<url or publication name | null>"}]\n\n'
    "Question:\n{question}\n"
)

def build_fact_check_task(agent) -> Task:
    return Task(
        description=FACT_CHECK_PROMPT,
        expected_output='[{"question":"string","answer":"string","source":"string or null"}]',
        agent=agent,
    )