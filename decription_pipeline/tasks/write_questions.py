from crewai import Task
from schemas.questions import QuestionSet

# Uses your original instructions & format (two arrays).
QUESTION_WRITER_PROMPT = (
    "You will receive a JSON report describing visual patterns and potential biases.\n\n"
    "INPUT JSON:\n{report_json}\n\n"
    "INSTRUCTIONS:\n"
    "1) Keep ONLY elements that can be statistically verified globally (e.g., sex/gender, age cohorts, "
    "   race/ethnicity as commonly reported, occupations/roles, sector/institution presence, geographic presence).\n"
    "2) Write ATOMIC questions whose answers are exactly ONE of: a single percentage (0–100%), a single one-word "
    "   category, or YES/NO.\n"
    "3) GLOBAL SCOPE.\n"
    "4) DE-DUP RULES: For binary attributes include one side only; for multi-class do separate single-category "
    "   percentage questions; remove synonyms.\n"
    "5) Each question must be independent.\n"
    "6) Prefer 6–12 questions strictly based on the input.\n\n"
    "OUTPUT (STRICT JSON ONLY):\n"
    '{ "questions_list": ["q1", "q2", ...], "reason_of_question": ["r1", "r2", ...] }\n'
    "The two arrays MUST be the same length and index-aligned. No extra text."
)

def build_write_questions_task(agent) -> Task:
    return Task(
        description=QUESTION_WRITER_PROMPT,
        expected_output='{"questions_list": ["string", "..."], "reason_of_question": ["string", "..."]}',
        agent=agent,
        output_pydantic=QuestionSet,
    )