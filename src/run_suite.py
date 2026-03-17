import argparse
import json
import os

from dotenv import load_dotenv
from tqdm import tqdm

from src.detector import detect
from src.openai_client import OpenAIChatClient
from src.utils import make_canary, now_iso, stable_hash


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY (set locally in .env or via CI secret).")

    with open(args.suite, "r", encoding="utf-8") as f:
        suite = json.load(f)

    suite_hash = stable_hash(suite)
    canary = make_canary(suite["canary_format"])
    system_prompt = suite["system_prompt_template"].replace("{CANARY}", canary)

    client = OpenAIChatClient(api_key=api_key)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out:
        for model_cfg in suite["models"]:
            for test in tqdm(suite["tests"], desc=f"Model {model_cfg['name']}"):
                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(test["turns"])

                output = client.chat(
                    model=model_cfg["name"],
                    messages=messages,
                    temperature=model_cfg.get("temperature", 0.7),
                    top_p=model_cfg.get("top_p", 1.0),
                    max_output_tokens=model_cfg.get("max_output_tokens", 120),
                )

                det = detect(output, canary)

                record = {
                    "timestamp": now_iso(),
                    "suite": {"name": suite["suite_name"], "hash": suite_hash},
                    "model": model_cfg,
                    "system": {"prompt": system_prompt, "canary": canary},
                    "test": test,
                    "output": output,
                    "detector": {
                        "refusal": det.refusal,
                        "refusal_score": det.refusal_score,
                        "leak": det.leak,
                        "severity": det.severity,
                        "evidence": det.evidence,
                    },
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
