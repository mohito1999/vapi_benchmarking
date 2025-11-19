import os
import requests
import time
import json
import argparse
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
 

# --- Configuration ---
load_dotenv()
VAPI_API_KEY = os.getenv("VAPI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

VAPI_BASE_URL = "https://api.vapi.ai"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# AGENT A (Current)
AGENT_A_PHONE_NUMBER = "+19847339851"
AGENT_A_PHONE_NUMBER_ID = "802718d4-04d4-43f8-a8ca-7bc21f20fb77"

# AGENT B (Proposed)
AGENT_B_PHONE_NUMBER = "+14436379041"
AGENT_B_PHONE_NUMBER_ID = "41d665b4-8f80-4efd-8d6e-f121fada5ae7"

# AGENT C (Caller)
AGENT_C_ASSISTANT_ID = "778ee932-963b-40e8-beb3-a240653e30f8"
AGENT_C_PHONE_NUMBER_ID = "3ab4d340-58b7-4eae-aa47-41942429f524"

BATCH_SIZE = 9
POLL_INTERVAL_SECONDS = 30
CALL_SETUP_WAIT_SECONDS = 20

# --- Audio Analysis ---
# Removed: Audio-Verified latency utilities (download, chunking, analysis).
# We now rely solely on Vapi-Reported latency for benchmarking.


# --- VAPI API ---
def make_vapi_call(number):
    try:
        r = requests.post(
            f"{VAPI_BASE_URL}/call/phone",
            headers={"Authorization": f"Bearer {VAPI_API_KEY}", "Content-Type": "application/json"},
            json={
                "assistantId": AGENT_C_ASSISTANT_ID,
                "phoneNumberId": AGENT_C_PHONE_NUMBER_ID,
                "customer": {"number": number},
            },
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"❌ Call initiation failed: {e}")
        return None


def find_inbound_call_id(phone_id, start_time):
    params = {"phoneNumberId": phone_id, "createdAtGt": start_time, "limit": 5}
    try:
        r = requests.get(
            f"{VAPI_BASE_URL}/call",
            headers={"Authorization": f"Bearer {VAPI_API_KEY}"},
            params=params,
            timeout=10,
        )
        r.raise_for_status()
        calls = sorted(
            [c for c in r.json() if c.get('type') == 'inboundPhoneCall'],
            key=lambda x: x['createdAt'],
            reverse=True,
        )
        return calls[0]['id'] if calls else None
    except Exception as e:
        print(f"❌ Call search failed: {e}")
        return None


def get_call_details(call_id):
    try:
        r = requests.get(
            f"{VAPI_BASE_URL}/call/{call_id}",
            headers={"Authorization": f"Bearer {VAPI_API_KEY}"},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"❌ Fetch details failed: {e}")
        return None


# --- Data Processing & Reporting ---
def parse_results(details):
    if not details:
        return {"success": False}

    llm, asr, tts = '?', '?', '?'
    for item in details.get('costs', []):
        if item.get('type') == 'model':
            llm = f"{item['model'].get('provider', '?')}/{item['model'].get('model', '?')}"
        elif item.get('type') == 'transcriber':
            asr = f"{item['transcriber'].get('provider', '?')}/{item['transcriber'].get('model', '?')}"
        elif item.get('type') == 'voice':
            tts = f"{item['voice'].get('provider', '?')}/{item['voice'].get('model', '?')}"

    perf = details.get("artifact", {}).get("performanceMetrics", {})

    score = None
    s = details.get("analysis", {}).get("successEvaluation")
    if s:
        try:
            score = [int(i) for i in s.split() if i.isdigit()][0]
        except (ValueError, IndexError):
            print(f"⚠️ Could not parse score: '{s}'")

    return {
        "id": details.get("id"),
        "cost": details.get("cost", 0),
        "vapi_latencies": perf.get("turnLatencies", []),
        "score": score,
        "llm": llm,
        "asr": asr,
        "tts": tts,
        "avg_comps": {
            "llm": perf.get("modelLatencyAverage"),
            "tts": perf.get("voiceLatencyAverage"),
            "asr": perf.get("transcriberLatencyAverage"),
        },
    }


def run_batch(targets):
    pending, active, completed = {}, {}, []
    print(f"🚀 Initiating batch of {len(targets)} calls...")
    for name, number, phone_id in targets:
        start_time = (datetime.now(timezone.utc) - timedelta(seconds=15)).isoformat().replace('+00:00', 'Z')
        call = make_vapi_call(number)
        if call:
            pending[call["id"]] = {"name": name, "phone_id": phone_id, "start": start_time}

    if not pending:
        print("No calls initiated.")
        return []

    print(f"⏳ Waiting {CALL_SETUP_WAIT_SECONDS}s...")
    time.sleep(CALL_SETUP_WAIT_SECONDS)

    print("🔍 Searching for inbound calls...")
    for out_id, info in pending.items():
        in_id = find_inbound_call_id(info['phone_id'], info['start'])
        if in_id:
            active[in_id] = info['name']
            print(f"  ✅ Found `{in_id}` for {info['name']}")
        else:
            print(f"  ❌ No match for {info['name']} (Outbound: {out_id})")

    if not active:
        print("No inbound calls to track.")
        return []

    with tqdm(total=len(active), desc="📊 Polling calls") as pbar:
        while active:
            time.sleep(POLL_INTERVAL_SECONDS)
            for in_id in list(active.keys()):
                details = get_call_details(in_id)
                if details and details.get("status") == "ended":
                    parsed = parse_results(details)
                    parsed["name"] = active.pop(in_id)
                    completed.append(parsed)
                    pbar.update(1)

    return completed


def generate_report(results_a, results_b):
    print("\n🧠 Summarizing data & generating report...")

    def get_stats(data):
        return {"mean": np.mean(data), "p50": np.percentile(data, 50), "p95": np.percentile(data, 95)} if data else {"mean": 'N/A', "p50": 'N/A', "p95": 'N/A'}

    def get_avg(res, comp):
        mean = np.mean([r['avg_comps'][comp] for r in res if r.get('avg_comps', {}).get(comp)])
        return 0 if math.isnan(mean) else mean

    summary = {}
    for cfg, res in [("A", results_a), ("B", results_b)]:
        v_lats = [t['turnLatency'] for r in res for t in r.get('vapi_latencies', [])]
        scores = [r['score'] for r in res if r.get('score')]
        summary[cfg] = {
            "name": f"Config {cfg}",
            "llm": res[0].get('llm', '?'),
            "asr": res[0].get('asr', '?'),
            "tts": res[0].get('tts', '?'),
            "vapi_stats_ms": {k: int(v) if isinstance(v, (int, float)) else v for k, v in get_stats(v_lats).items()},
            "avg_comps_ms": {"llm": int(get_avg(res, 'llm')), "tts": int(get_avg(res, 'tts')), "asr": int(get_avg(res, 'asr'))},
            "avg_score": round(np.mean(scores), 2) if scores else 'N/A',
            "total_cost": round(sum(r.get('cost', 0) for r in res), 4),
        }

    summary_data = summary

    prompt = f"""You are an expert AI Performance Analyst. Write a single markdown report comparing two voice agent configurations using the JSON data below.

Context:
- Only use Vapi-reported latency metrics provided in the data. Do not use or reference audio-verified latency.
- Explain each metric in plain English: what 'Vapi-Reported Turn Latency' means, what the component latencies (LLM, Voice/TTS, ASR) represent, and how they relate to user-perceived speed.
- P95 latency is the primary indicator of speed and reliability under real-world conditions; Mean and Median are secondary.
- Do not bias the recommendation. Examine the data and justify which configuration is better and why, or state if results are inconclusive.

Data:
```json
{json.dumps(summary_data, indent=2)}
Instructions:
- Title and Intro: Start with a clear title. In the introduction, describe each configuration using its llm, asr, and tts fields, and briefly define the metrics in layman's terms.
- Executive Summary: Provide a neutral, data-driven recommendation (or note if inconclusive), primarily using Vapi-reported P95 latency, supported by component breakdowns and effectiveness.
- Latency Analysis Section: Create a markdown table for Vapi-Reported Latency (Mean, Median [p50], P95) for each config.
- Component Latency Breakdown: Create a table for average component latencies (LLM, Voice/TTS, ASR) to explain potential causes of differences.
- Effectiveness & Cost Table: Show average score and total cost.
- Conclusion: Summarize findings and the reasoning in simple terms.
- Strictly use the provided data. Do not invent metrics.
- Output must be a single block of markdown with no preamble or extra commentary.
"""

    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "openrouter/sherlock-think-alpha", "messages": [{"role": "user", "content": prompt}]}

    try:
        r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=90)
        r.raise_for_status()
        summary_report = r.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"\n❌ LLM summary failed: {e}")
        summary_report = "# Report Failed"

    details_md = "\n\n---\n\n## Detailed Call Metrics\nThis provides a turn-by-turn breakdown for each call.\n"

    def fmt_details(res, name):
        m = res[0] if res else {}
        section = f"\n### {name} (LLM: {m.get('llm','?')} | ASR: {m.get('asr','?')} | TTS: {m.get('tts','?')})\n"
        for i, r in enumerate(res):
            section += f"\nCall {i+1} ({r['id']})\n- Score: {r.get('score','N/A')} | Cost: ${r.get('cost',0):.4f}\n\n"
            section += "| Turn | Vapi Turn Latency (ms) | LLM (ms) | TTS (ms) | ASR (ms) |\n|:----:|:-----------------------:|:----------:|:----------:|:--------:|\n"
            v = r.get("vapi_latencies", [])
            if not v:
                section += "| N/A | N/A | N/A | N/A | N/A |\n"
                continue
            for j in range(len(v)):
                vt = v[j]
                section += f"| {j+1} | {vt.get('turnLatency','N/A')} | {vt.get('modelLatency','N/A')} | {vt.get('voiceLatency','N/A')} | {vt.get('transcriberLatency','N/A')} |\n"
        return section

    details_md += fmt_details(results_a, "Config A")
    details_md += fmt_details(results_b, "Config B")

    plt.figure(figsize=(8, 5))
    v_a = [t['turnLatency'] for r in results_a for t in r.get('vapi_latencies', [])]
    v_b = [t['turnLatency'] for r in results_b for t in r.get('vapi_latencies', [])]
    data_v = [d for d in [v_a, v_b] if d]
    labels_v = [n for d, n in zip([v_a, v_b], ['A', 'B']) if d]
    if data_v:
        plt.boxplot(data_v, labels=labels_v)
    plt.title('Vapi-Reported Latency')
    plt.ylabel('ms')
    plt.grid(True, alpha=0.6)

    plt.tight_layout()
    plot_filename = 'latency_comparison.png'
    plt.savefig(plot_filename)
    print(f"\n📈 Chart saved: {plot_filename}")

    final_report = summary_report + details_md
    with open("benchmark_report.md", "w") as f:
        f.write(final_report)
    print("✅ Report saved: benchmark_report.md")


def main():
    parser = argparse.ArgumentParser(description="Run Vapi agent benchmark.")
    parser.add_argument("--test", action="store_true", help="Run in test mode (1 call per agent).")
    args = parser.parse_args()

    if not VAPI_API_KEY or not OPENROUTER_API_KEY:
        print("❌ API keys not in .env file.")
        return

    # Prepare targets
    targets = [
        ("Agent A", AGENT_A_PHONE_NUMBER, AGENT_A_PHONE_NUMBER_ID)
    ] * (1 if args.test else 50) + [
        ("Agent B", AGENT_B_PHONE_NUMBER, AGENT_B_PHONE_NUMBER_ID)
    ] * (1 if args.test else 50)

    all_results = []
    for i in range(0, len(targets), BATCH_SIZE):
        batch_targets = targets[i:i + BATCH_SIZE]
        batch_results = run_batch(batch_targets)
        all_results.extend(batch_results)

    results_a = [r for r in all_results if r.get("name") == "Agent A"]
    results_b = [r for r in all_results if r.get("name") == "Agent B"]

    print(f"\n--- Benchmark Complete ---\nTotal calls processed: {len(all_results)}")
    if not results_a or not results_b:
        print("\n❌ No successful calls to report.")
        return

    generate_report(results_a, results_b)


if __name__ == "__main__":
    main()