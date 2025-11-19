import os
import requests
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv
import difflib

load_dotenv()

VAPI_BASE_URL = "https://api.vapi.ai"
VAPI_API_KEY = os.getenv("VAPI_API_KEY")

def get_assistant_details(assistant_id):
    """Fetches assistant details from Vapi API."""
    if not assistant_id:
        return None
        
    try:
        url = f"{VAPI_BASE_URL}/assistant/{assistant_id}"
        headers = {"Authorization": f"Bearer {VAPI_API_KEY}"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Failed to fetch assistant {assistant_id}: {e}")
        return None

def clean_for_diff(data):
    """Removes dynamic fields like 'createdAt', 'updatedAt', 'id' to focus on config."""
    if isinstance(data, dict):
        return {k: clean_for_diff(v) for k, v in data.items() if k not in ['id', 'createdAt', 'updatedAt', 'orgId']}
    elif isinstance(data, list):
        return [clean_for_diff(i) for i in data]
    else:
        return data

def generate_diff(assistant_a, assistant_b):
    """Generates a diff between two assistant configurations."""
    if not assistant_a or not assistant_b:
        return None

    a_clean = clean_for_diff(assistant_a)
    b_clean = clean_for_diff(assistant_b)

    a_str = json.dumps(a_clean, indent=2, sort_keys=True).splitlines()
    b_str = json.dumps(b_clean, indent=2, sort_keys=True).splitlines()

    diff = difflib.unified_diff(
        a_str, b_str, 
        fromfile=f"Agent A ({assistant_a.get('name', 'Unknown')})", 
        tofile=f"Agent B ({assistant_b.get('name', 'Unknown')})", 
        lineterm=""
    )
    
    return list(diff)

def format_diff_report(diff_lines, assistant_a, assistant_b):
    """Formats the diff output into a readable Markdown report."""
    if not diff_lines:
        return "## No differences found between the two assistants."

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"# Assistant Configuration Diff Report\n"
    report += f"**Generated:** {timestamp}\n\n"
    
    report += f"## Agents Compared\n"
    report += f"- **Agent A**: {assistant_a.get('name', 'Unknown')} (ID: `{assistant_a.get('id')}`)\n"
    report += f"- **Agent B**: {assistant_b.get('name', 'Unknown')} (ID: `{assistant_b.get('id')}`)\n\n"
    
    report += "## Configuration Differences\n"
    report += "```diff\n"
    for line in diff_lines:
        report += line + "\n"
    report += "```\n"
    
    return report

def save_report(content):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"diff_report_{timestamp}.md"
    with open(filename, "w") as f:
        f.write(content)
    print(f"✅ Diff report saved: {filename}")
    return filename

def main():
    parser = argparse.ArgumentParser(description="Compare two Vapi assistants.")
    parser.add_argument("--id1", help="ID of the first assistant (Agent A)")
    parser.add_argument("--id2", help="ID of the second assistant (Agent B)")
    args = parser.parse_args()

    id1 = args.id1 or os.getenv("AGENT_A_ID")
    id2 = args.id2 or os.getenv("AGENT_B_ID")

    if not VAPI_API_KEY:
        print("❌ VAPI_API_KEY not found in environment variables.")
        return

    if not id1 or not id2:
        print("❌ Please provide both assistant IDs via arguments or .env (AGENT_A_ID, AGENT_B_ID).")
        return

    print(f"🔍 Fetching details for Agent A ({id1})...")
    a_data = get_assistant_details(id1)
    
    print(f"🔍 Fetching details for Agent B ({id2})...")
    b_data = get_assistant_details(id2)

    if a_data and b_data:
        print("⚖️  Comparing configurations...")
        diff_lines = generate_diff(a_data, b_data)
        report = format_diff_report(diff_lines, a_data, b_data)
        save_report(report)
    else:
        print("❌ Could not fetch details for one or both agents.")

if __name__ == "__main__":
    main()
