# Experiment for research question 1:
# how does the way a prompt is written change the answer quality across models?
# I give every model the same task in three prompt styles and save the results.
#
# Run with: python experiment.py llama3.2:3b qwen2.5:3b

import csv
import sys
import time

import httpx

# Ollama runs locally on this port
OLLAMA = "http://localhost:11434"

# the task stays the same, only the prompt style changes
TASK_TEXT = (
    "Our company moved its backup schedule from nightly to weekly to save storage costs, "
    "but two customers lost a week of data after a disk failure."
)

PROMPTS = {
    # almost no instructions, to see what the model does by default
    "vague": f"Explain this problem: {TASK_TEXT}",

    # gives a role, an audience and a fixed output format
    "detailed": (
        "You are an IT risk consultant. Explain the following incident to a non-technical manager "
        "in exactly 3 short bullet points: what happened, why it happened, how to prevent it.\n\n"
        f"Incident: {TASK_TEXT}"
    ),

    # shows one worked example so the model can copy the pattern
    "few_shot": (
        "Rewrite incidents as: CAUSE -> IMPACT -> FIX.\n\n"
        "Example:\nIncident: A server ran out of disk space and crashed.\n"
        "CAUSE: no disk monitoring -> IMPACT: 2h outage -> FIX: add disk alerts at 80%.\n\n"
        f"Incident: {TASK_TEXT}"
    ),
}


def run(model, prompt):
    # send one prompt to Ollama and measure how long the full answer takes
    start = time.time()
    r = httpx.post(
        f"{OLLAMA}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=300,
    )
    return r.json()["response"], time.time() - start


if __name__ == "__main__":
    # models come from the command line, llama3.2:3b is the default
    models = sys.argv[1:] or ["llama3.2:3b"]

    with open("results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "prompt_style", "seconds", "chars", "response"])

        # every model gets every prompt style
        for m in models:
            for style, p in PROMPTS.items():
                print(f"Running {m} / {style} ...")
                text, secs = run(m, p)
                w.writerow([m, style, round(secs, 1), len(text), text])

    print("Done, results saved in results.csv")
