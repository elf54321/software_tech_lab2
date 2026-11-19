# experiment v2: warm-up call, 3 repeats, 3 tasks
# run with: python experiment.py llama3.2:3b qwen2.5:3b

import csv
import statistics
import sys
import time

import httpx

OLLAMA = "http://localhost:11434"
REPEATS = 3

TASKS = {
    "backup": "Our company moved its backup schedule from nightly to weekly to save storage costs, "
              "but two customers lost a week of data after a disk failure.",
    "password": "A company forces employees to change their password every 30 days, "
                "and now many employees write their passwords on sticky notes.",
    "meetings": "A team holds a one-hour meeting every day, "
                "and the developers say they have no time left to write code.",
}


def styles(task):
    return {
        "vague": f"Explain this problem: {task}",
        "detailed": (
            "You are an IT risk consultant. Explain the following incident to a non-technical manager "
            "in exactly 3 short bullet points: what happened, why it happened, how to prevent it.\n\n"
            f"Incident: {task}"
        ),
        "few_shot": (
            "Rewrite incidents as: CAUSE -> IMPACT -> FIX.\n\n"
            "Example:\nIncident: A server ran out of disk space and crashed.\n"
            "CAUSE: no disk monitoring -> IMPACT: 2h outage -> FIX: add disk alerts at 80%.\n\n"
            f"Incident: {task}"
        ),
    }


def run(model, prompt):
    start = time.time()
    r = httpx.post(
        f"{OLLAMA}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False,
              "options": {"temperature": 0.7}},
        timeout=300,
    )
    return r.json()["response"], time.time() - start


if __name__ == "__main__":
    models = sys.argv[1:] or ["llama3.2:3b"]
    rows = []

    for m in models:
        print(f"Warm-up {m} ...")
        run(m, "Say hello.")
        for task_name, task in TASKS.items():
            for style, prompt in styles(task).items():
                for i in range(1, REPEATS + 1):
                    print(f"{m} / {task_name} / {style} / run {i}")
                    text, secs = run(m, prompt)
                    rows.append([m, task_name, style, i, round(secs, 1), len(text), text])

    with open("results_v2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "task", "style", "run", "seconds", "chars", "response"])
        w.writerows(rows)

    print("\nAverages (seconds, characters):")
    for m in models:
        for style in ["vague", "detailed", "few_shot"]:
            sel = [r for r in rows if r[0] == m and r[2] == style]
            print(f"{m:12} {style:9} {statistics.mean(r[4] for r in sel):5.1f}s "
                  f"{statistics.mean(r[5] for r in sel):7.0f} chars")
