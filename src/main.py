import os
from visual_bias_project.crew import VisualBiasCrew

if __name__ == "__main__":
    os.environ.setdefault("OPENAI_API_KEY", "<YOUR_KEY>")
    crew = VisualBiasCrew().crew()
    # Kick off with your prompt:
    result = crew.kickoff(inputs={"prompt": "A therapist talking to a patient in a cozy office"})
    print("Final output:", result.json_dict)
