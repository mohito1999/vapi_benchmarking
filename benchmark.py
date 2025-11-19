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
import dateutil.parser

# --- Configuration ---
load_dotenv()
VAPI_API_KEY = os.getenv("VAPI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

VAPI_BASE_URL = "https://api.vapi.ai"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# AGENT A (Current)
AGENT_A_PHONE_NUMBER = os.getenv("AGENT_A_PHONE_NUMBER", "+19847339851")
AGENT_A_PHONE_NUMBER_ID = os.getenv("AGENT_A_PHONE_NUMBER_ID", "802718d4-04d4-43f8-a8ca-7bc21f20fb77")

# AGENT B (Proposed)
AGENT_B_PHONE_NUMBER = os.getenv("AGENT_B_PHONE_NUMBER", "+14436379041")
AGENT_B_PHONE_NUMBER_ID = os.getenv("AGENT_B_PHONE_NUMBER_ID", "41d665b4-8f80-4efd-8d6e-f121fada5ae7")

# AGENT C (Caller)
AGENT_C_ASSISTANT_ID = os.getenv("AGENT_C_ASSISTANT_ID", "778ee932-963b-40e8-beb3-a240653e30f8")
AGENT_C_PHONE_NUMBER_ID = os.getenv("AGENT_C_PHONE_NUMBER_ID", "80c50b31-b903-40d4-bd11-fd278dfa9711")

# Default Batch Size 5 to respect 10 concurrency limit (5 outbound + 5 inbound)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 5))
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", 10))
CALL_SETUP_WAIT_SECONDS = 20
MIN_DURATION_SECONDS = 45

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

    # Calculate Duration
    started_at_str = details.get("startedAt")
    ended_at_str = details.get("endedAt")
    duration = 0
    if started_at_str and ended_at_str:
        try:
            # Handle Z format manually if needed, or use dateutil
            # Vapi returns ISO 8601 like "2025-11-19T04:53:33.747Z"
            start = dateutil.parser.isoparse(started_at_str)
            end = dateutil.parser.isoparse(ended_at_str)
            duration = (end - start).total_seconds()
        except Exception as e:
            print(f"⚠️ Error parsing dates for {details.get('id')}: {e}")

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
            pass

    # Calculate average endpointing latency if not provided
    endpointing_avg = perf.get("endpointingLatencyAverage")
    if endpointing_avg is None:
        lats = [t.get('endpointingLatency') for t in perf.get("turnLatencies", []) if t.get('endpointingLatency') is not None]
        if lats:
            endpointing_avg = sum(lats) / len(lats)

    return {
        "id": details.get("id"),
        "cost": details.get("cost", 0),
        "vapi_latencies": perf.get("turnLatencies", []),
        "score": score,
        "llm": llm,
        "asr": asr,
        "tts": tts,
        "duration": duration,
        "avg_comps": {
            "llm": perf.get("modelLatencyAverage"),
            "tts": perf.get("voiceLatencyAverage"),
            "asr": perf.get("transcriberLatencyAverage"),
            "endpointing": endpointing_avg,
        },
    }


def run_strict_batch(targets, batch_num):
    """
    Executes a batch of calls using Strict Batching:
    1. Fire all calls.
    2. Wait.
    3. Find all resulting inbound calls.
    4. Wait for ALL to complete.
    """
    print(f"\n🚀 Starting Batch {batch_num} ({len(targets)} calls)...")
    
    # 1. Capture start time (with a small buffer)
    batch_start_time = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat().replace('+00:00', 'Z')
    
    # 2. Initiate Calls
    outbound_map = {} # phone_number -> agent_name
    agent_phone_ids = {} # phone_number -> phone_id
    
    for name, number, phone_id in targets:
        make_vapi_call(number)
        outbound_map[number] = name
        agent_phone_ids[number] = phone_id
        # Small delay to avoid rate limits if any
        time.sleep(0.2)

    print(f"⏳ Waiting {CALL_SETUP_WAIT_SECONDS}s for calls to connect...")
    time.sleep(CALL_SETUP_WAIT_SECONDS)

    # 3. Find Inbound Calls
    # We need to query for each unique agent phone number involved in this batch
    unique_phone_ids = set(agent_phone_ids.values())
    active_calls = {} # inbound_call_id -> agent_name

    print("🔍 Finding inbound calls...")
    found_count = 0
    for pid in unique_phone_ids:
        # We expect roughly N calls for this ID in this batch
        # We fetch a bit more than batch size to be safe, but filter by time
        
        params = {"phoneNumberId": pid, "createdAtGt": batch_start_time, "limit": BATCH_SIZE * 2}
        try:
            r = requests.get(
                f"{VAPI_BASE_URL}/call",
                headers={"Authorization": f"Bearer {VAPI_API_KEY}"},
                params=params,
                timeout=10,
            )
            r.raise_for_status()
            calls = [c for c in r.json() if c.get('type') == 'inboundPhoneCall']
        except Exception as e:
            print(f"❌ Call search failed for {pid}: {e}")
            calls = []
        
        for c in calls:
            if c['status'] != 'ended': 
                 # Map PID back to name
                 agent_name = next((name for name, num, p_id in targets if p_id == pid), "Unknown")
                 active_calls[c['id']] = agent_name
                 found_count += 1

    print(f"  ✅ Found {found_count} active calls. Waiting for completion...")

    if not active_calls:
        print("  ⚠️ No active calls found. Moving to next batch.")
        return []

    # 4. Poll until ALL are done
    completed_results = []
    
    while active_calls:
        time.sleep(POLL_INTERVAL_SECONDS)
        
        for call_id in list(active_calls.keys()):
            details = get_call_details(call_id)
            if details and details.get("status") == "ended":
                agent_name = active_calls.pop(call_id)
                parsed = parse_results(details)
                parsed["name"] = agent_name
                completed_results.append(parsed)
                print(f"    Call {call_id} finished ({len(active_calls)} remaining)")
            elif not details:
                pass
    
    return completed_results


def generate_report(results_a, results_b, context_str=""):
    print("\n🧠 Summarizing data & generating report...")

    # Filter results
    # Criteria: Duration >= MIN_DURATION_SECONDS AND Turns > 2
    def is_valid(r):
        turns = len(r.get('vapi_latencies', []))
        return r['duration'] >= MIN_DURATION_SECONDS and turns > 2

    def get_exclusion_reason(r):
        reasons = []
        if r['duration'] < MIN_DURATION_SECONDS:
            reasons.append(f"Duration < {MIN_DURATION_SECONDS}s")
        if len(r.get('vapi_latencies', [])) <= 2:
            reasons.append("Turns <= 2")
        return ", ".join(reasons) if reasons else "Unknown"

    valid_a = [r for r in results_a if is_valid(r)]
    excluded_a = [r for r in results_a if not is_valid(r)]
    
    valid_b = [r for r in results_b if is_valid(r)]
    excluded_b = [r for r in results_b if not is_valid(r)]
    
    all_excluded = excluded_a + excluded_b

    def get_stats(data):
        return {"mean": np.mean(data), "p50": np.percentile(data, 50), "p95": np.percentile(data, 95)} if data else {"mean": 'N/A', "p50": 'N/A', "p95": 'N/A'}

    def get_avg(res, comp):
        vals = [r['avg_comps'][comp] for r in res if r.get('avg_comps', {}).get(comp) is not None]
        return np.mean(vals) if vals else 0

    summary = {}
    for cfg, res in [("A", valid_a), ("B", valid_b)]:
        if not res:
            summary[cfg] = {"name": f"Config {cfg}", "error": "No Valid Data (All calls excluded?)"}
            continue
            
        v_lats = [t['turnLatency'] for r in res for t in r.get('vapi_latencies', [])]
        scores = [r['score'] for r in res if r.get('score') is not None]
        
        summary[cfg] = {
            "name": f"Config {cfg}",
            "llm": res[0].get('llm', '?'),
            "asr": res[0].get('asr', '?'),
            "tts": res[0].get('tts', '?'),
            "vapi_stats_ms": {k: int(v) if isinstance(v, (int, float)) else v for k, v in get_stats(v_lats).items()},
            "avg_comps_ms": {
                "llm": int(get_avg(res, 'llm')), 
                "tts": int(get_avg(res, 'tts')), 
                "asr": int(get_avg(res, 'asr')),
                "endpointing": int(get_avg(res, 'endpointing'))
            },
            "avg_score": round(np.mean(scores), 2) if scores else 'N/A',
            "total_cost": round(sum(r.get('cost', 0) for r in res), 4),
            "call_count": len(res)
        }

    summary_data = summary

    prompt = f"""You are an expert AI Performance Analyst. Write a single markdown report comparing two voice agent configurations using the JSON data below.

    Context:
    - User provided context for this run: "{context_str}"
    - Only use Vapi-reported latency metrics provided in the data.
    - **Analysis Logic**: P95 is the primary indicator for tail-end speed, but you must weigh it against P50 (consistency) and Sample Size (reliability).
    - **Quality Check**: Critically evaluate if lower latency came at the cost of lower Average Scores (e.g., due to interruptions).
    - Note that calls shorter than {MIN_DURATION_SECONDS}s OR with <= 2 turns have been excluded.

    Data:
    ```json
    {json.dumps(summary_data, indent=2)}
    Instructions (Strictly follow this structure):
    Title and Introduction: Describe configs, metrics, exclusion criteria, and user context.
    Executive Summary: Provide a balanced recommendation. Do not recommend a configuration solely on speed if the sample size is insignificant or if quality scores dropped.
    Latency Analysis: Table for Vapi-Reported Latency (Mean, P50, P95). Briefly analyze the spread between P50 and P95.
    Component Breakdown: Table for LLM, TTS, ASR, and Endpointing latencies. Highlight which component drove the biggest change.
    Effectiveness & Cost: Table for scores, total cost, and call count.
    Conclusion: Summarize trade-offs between speed, cost, and reliability.
    Output markdown only.
    """

    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "openrouter/sherlock-think-alpha", "messages": [{"role": "user", "content": prompt}]}

    try:
        r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=90)
        r.raise_for_status()
        summary_report = r.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"\n❌ LLM summary failed: {e}")
        summary_report = "# Report Failed\nCould not generate summary."

    # Append Excluded Calls Table
    excluded_md = f"\n\n## ⚠️ Excluded Calls (<{MIN_DURATION_SECONDS}s or <=2 turns)\n"
    if all_excluded:
        excluded_md += "| ID | Config | Duration (s) | Turns | Reason |\n|---|---|---|---|---|\n"
        for r in all_excluded:
            reason = get_exclusion_reason(r)
            turns = len(r.get('vapi_latencies', []))
            excluded_md += f"| `{r['id']}` | {r['name']} | {r['duration']:.2f} | {turns} | {reason} |\n"
    else:
        excluded_md += "No calls were excluded.\n"

    details_md = "\n\n---\n\n## Detailed Call Metrics (Valid Calls)\n"

    def fmt_details(res, name):
        if not res: return f"\n### {name}\nNo valid data.\n"
        m = res[0]
        section = f"\n### {name} (LLM: {m.get('llm','?')} | ASR: {m.get('asr','?')} | TTS: {m.get('tts','?')})\n"
        for i, r in enumerate(res):
            section += f"\n**Call {i+1}** (ID: `{r['id']}`) - Duration: {r['duration']:.2f}s\n- Score: {r.get('score','N/A')} | Cost: ${r.get('cost',0):.4f}\n\n"
            section += "| Turn | Latency (ms) | LLM | TTS | ASR | Endpointing |\n|:--:|:--:|:--:|:--:|:--:|:--:|\n"
            v = r.get("vapi_latencies", [])
            if not v:
                section += "| - | - | - | - | - | - |\n"
                continue
            for j, vt in enumerate(v):
                section += f"| {j+1} | {vt.get('turnLatency','-')} | {vt.get('modelLatency','-')} | {vt.get('voiceLatency','-')} | {vt.get('transcriberLatency','-')} | {vt.get('endpointingLatency','-')} |\n"
        return section

    details_md += fmt_details(valid_a, "Config A")
    details_md += fmt_details(valid_b, "Config B")

    # Plotting (Only Valid Calls)
    try:
        plt.figure(figsize=(8, 5))
        v_a = [t['turnLatency'] for r in valid_a for t in r.get('vapi_latencies', [])]
        v_b = [t['turnLatency'] for r in valid_b for t in r.get('vapi_latencies', [])]
        data_v = [d for d in [v_a, v_b] if d]
        labels_v = [n for d, n in zip([v_a, v_b], ['A', 'B']) if d]
        if data_v:
            plt.boxplot(data_v, labels=labels_v)
            plt.title(f'Vapi-Reported Latency (Valid Calls)')
            plt.ylabel('Latency (ms)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('latency_comparison.png')
            print(f"\n📈 Chart saved: latency_comparison.png")
    except Exception as e:
        print(f"⚠️ Chart generation failed: {e}")

    # Timestamped Filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_report_{timestamp}.md"
    
    final_report = summary_report + excluded_md + details_md
    with open(filename, "w") as f:
        f.write(final_report)
    print(f"✅ Report saved: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Run Vapi agent benchmark.")
    parser.add_argument("--test", action="store_true", help="Run in test mode (1 call per agent).")
    parser.add_argument("--calls", type=int, help="Number of calls per agent configuration.")
    parser.add_argument("--context", type=str, default="", help="Context for the report (e.g. description of changes).")
    args = parser.parse_args()

    if not VAPI_API_KEY or not OPENROUTER_API_KEY:
        print("❌ API keys not in .env file.")
        return

    # Determine number of calls
    if args.calls:
        num_calls = args.calls
    elif args.test:
        num_calls = 1
    else:
        num_calls = 50

    print(f"📋 Configuration: {num_calls} calls per agent, Batch Size: {BATCH_SIZE}")
    if args.context:
        print(f"📝 Context: {args.context}")

    targets = []
    targets.extend([("Agent A", AGENT_A_PHONE_NUMBER, AGENT_A_PHONE_NUMBER_ID)] * num_calls)
    targets.extend([("Agent B", AGENT_B_PHONE_NUMBER, AGENT_B_PHONE_NUMBER_ID)] * num_calls)

    all_results = []
    
    # Process in batches
    total_batches = math.ceil(len(targets) / BATCH_SIZE)
    
    for i in range(total_batches):
        batch_targets = targets[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        batch_res = run_strict_batch(batch_targets, i + 1)
        all_results.extend(batch_res)

    results_a = [r for r in all_results if r.get("name") == "Agent A"]
    results_b = [r for r in all_results if r.get("name") == "Agent B"]

    print(f"\n--- Benchmark Complete ---\nTotal calls processed: {len(all_results)}")
    
    if not results_a and not results_b:
        print("\n❌ No successful calls to report.")
        return

    generate_report(results_a, results_b, args.context)


if __name__ == "__main__":
    main()