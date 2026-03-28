import argparse
import json
import os

from dotenv import load_dotenv
from tqdm import tqdm

from src.detector import detect, DetectionResult
from src.openai_client import OpenAIChatClient
from src.anthropic_client import AnthropicChatClient
from src.utils import make_canary, now_iso, stable_hash

_SEV_ORDER = {"L0": 0, "L2": 1, "L3": 2, "L4": 3}


def worst_detection(results: list[DetectionResult]) -> DetectionResult:
    return max(results, key=lambda d: _SEV_ORDER.get(d.severity, 0))


def run_single_turn(client, model_cfg, system_prompt, turns, canary):
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(turns)

    output = client.chat(
        model=model_cfg["name"],
        messages=messages,
        temperature=model_cfg.get("temperature", 0.7),
        top_p=model_cfg.get("top_p", 1.0),
        max_output_tokens=model_cfg.get("max_output_tokens", 120),
    )
    det = detect(output, canary)
    return output, [output], det


def run_multi_turn(client, model_cfg, system_prompt, turns, canary):
    messages = [{"role": "system", "content": system_prompt}]
    all_outputs = []
    all_detections = []

    for turn in turns:
        messages.append(turn)

        output = client.chat(
            model=model_cfg["name"],
            messages=messages,
            temperature=model_cfg.get("temperature", 0.7),
            top_p=model_cfg.get("top_p", 1.0),
            max_output_tokens=model_cfg.get("max_output_tokens", 120),
        )

        messages.append({"role": "assistant", "content": output})
        all_outputs.append(output)
        all_detections.append(detect(output, canary))

    full_output = "\n---\n".join(all_outputs)
    best_det = worst_detection(all_detections)
    return full_output, all_outputs, best_det


def build_clients():
    """Build one client per provider from available env vars."""
    clients = {}

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        clients["openai"] = OpenAIChatClient(api_key=openai_key)

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        clients["anthropic"] = AnthropicChatClient(api_key=anthropic_key)

    return clients


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    args = p.parse_args()

    load_dotenv()

    with open(args.suite, "r", encoding="utf-8") as f:
        suite = json.load(f)

    tests = suite["tests"]
    tests = [t for i, t in enumerate(tests) if i % args.shards == args.shard]

    suite_hash = stable_hash(suite)
    canary = make_canary(suite["canary_format"])

    role_prompts = suite.get("system_prompt_roles", {})
    legacy_template = suite.get("system_prompt_template")

    def get_system_prompt(role: str) -> str:
        if role in role_prompts:
            return role_prompts[role].replace("{CANARY}", canary)
        if legacy_template:
            return legacy_template.replace("{CANARY}", canary)
        raise ValueError(f"No system prompt found for role '{role}'")

    clients = build_clients()
    if not clients:
        raise RuntimeError(
            "No API keys found. Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY in .env"
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out:
        for model_cfg in suite["models"]:
            provider = model_cfg.get("provider", "openai")

            if provider not in clients:
                print(
                    f"⚠️  Skipping {model_cfg['name']}: "
                    f"no API key for provider '{provider}'. "
                    f"Set {provider.upper()}_API_KEY in .env to enable."
                )
                continue

            client = clients[provider]

            for test in tqdm(tests, desc=f"{model_cfg['name']}"):
                role = test.get("role", "support")
                system_prompt = get_system_prompt(role)
                is_multi = test["family"].startswith("multi_turn/")

                if is_multi:
                    full_output, turn_outputs, det = run_multi_turn(
                        client, model_cfg, system_prompt, test["turns"], canary
                    )
                else:
                    full_output, turn_outputs, det = run_single_turn(
                        client, model_cfg, system_prompt, test["turns"], canary
                    )

                record = {
                    "timestamp": now_iso(),
                    "suite": {"name": suite["suite_name"], "hash": suite_hash},
                    "model": model_cfg,
                    "system": {
                        "role": role,
                        "prompt": system_prompt,
                        "canary": canary,
                    },
                    "test": test,
                    "output": full_output,
                    "turn_outputs": turn_outputs,
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