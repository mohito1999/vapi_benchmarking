import streamlit as st
import subprocess
import os
import time
import re
import glob
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Vapi Agent Benchmark", layout="wide")

st.title("🤖 Vapi Agent Benchmark")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Configuration")
    
    # API Keys
    vapi_key = st.text_input("Vapi API Key", value=os.getenv("VAPI_API_KEY", ""), type="password")
    openrouter_key = st.text_input("OpenRouter API Key", value=os.getenv("OPENROUTER_API_KEY", ""), type="password")
    
    st.divider()
    
    # Agent A
    st.subheader("Agent A (Control)")
    agent_a_phone = st.text_input("Phone Number", value=os.getenv("AGENT_A_PHONE_NUMBER", "+19847339851"))
    agent_a_id = st.text_input("Phone ID", value=os.getenv("AGENT_A_PHONE_NUMBER_ID", "802718d4-04d4-43f8-a8ca-7bc21f20fb77"))
    
    st.divider()
    
    # Agent B
    st.subheader("Agent B (Test)")
    agent_b_phone = st.text_input("Phone Number", value=os.getenv("AGENT_B_PHONE_NUMBER", "+14436379041"))
    agent_b_id = st.text_input("Phone ID", value=os.getenv("AGENT_B_PHONE_NUMBER_ID", "41d665b4-8f80-4efd-8d6e-f121fada5ae7"))
    
    st.divider()
    
    # Agent C (Caller)
    st.subheader("Caller Agent")
    agent_c_asst_id = st.text_input("Assistant ID", value=os.getenv("AGENT_C_ASSISTANT_ID", "778ee932-963b-40e8-beb3-a240653e30f8"))
    agent_c_phone_id = st.text_input("Phone ID", value=os.getenv("AGENT_C_PHONE_NUMBER_ID", "3ab4d340-58b7-4eae-aa47-41942429f524"))

    st.divider()
    
    st.header("Run Settings")
    num_calls = st.number_input("Calls per Agent", min_value=1, max_value=100, value=5)
    batch_size = st.slider("Batch Size", min_value=1, max_value=10, value=5)

# --- Main Execution ---

context = st.text_area("Report Context / Configuration Differences", placeholder="e.g., Config B uses a new system prompt with lower latency settings...")

if st.button("🚀 Run Benchmark", type="primary"):
    if not vapi_key or not openrouter_key:
        st.error("Please provide both API keys.")
        st.stop()

    # Save config to temporary .env or pass via env vars
    env = os.environ.copy()
    env["VAPI_API_KEY"] = vapi_key
    env["OPENROUTER_API_KEY"] = openrouter_key
    
    # Update env vars for the subprocess
    env["AGENT_A_PHONE_NUMBER"] = agent_a_phone
    env["AGENT_A_PHONE_NUMBER_ID"] = agent_a_id
    env["AGENT_B_PHONE_NUMBER"] = agent_b_phone
    env["AGENT_B_PHONE_NUMBER_ID"] = agent_b_id
    env["AGENT_C_ASSISTANT_ID"] = agent_c_asst_id
    env["AGENT_C_PHONE_NUMBER_ID"] = agent_c_phone_id
    env["BATCH_SIZE"] = str(batch_size)

    # Progress UI
    progress_bar = st.progress(0)
    status_text = st.empty()
    log_area = st.empty()
    
    logs = []
    
    def update_logs(line):
        logs.append(line)
        # Keep only last 15 lines for cleaner UI
        log_text = "\n".join(logs[-15:])
        log_area.code(log_text, language="text")

    # Run the script
    cmd = ["python", "-u", "benchmark.py", "--calls", str(num_calls), "--context", context]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
            universal_newlines=True
        )
        
        # Monitor output
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                line = line.strip()
                if line: # Skip empty lines
                    update_logs(line)

        if process.returncode == 0:
            st.success("Benchmark Complete!")
            
            # Find the most recent report file
            try:
                list_of_files = glob.glob('benchmark_report_*.md') 
                if list_of_files:
                    latest_file = max(list_of_files, key=os.path.getctime)
                    
                    with open(latest_file, "r") as f:
                        report_content = f.read()
                    
                    st.markdown(report_content)
                    
                    if os.path.exists("latency_comparison.png"):
                        st.image("latency_comparison.png", caption="Latency Distribution")
                else:
                    st.warning("No report file found.")
            except Exception as e:
                st.error(f"Error loading report: {e}")
                
        else:
            st.error("Benchmark failed. Check logs.")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
