# Vapi Agent Benchmark Tool

A powerful Streamlit-based tool for benchmarking and comparing the performance of Vapi voice agents. This tool allows you to run concurrent calls, analyze latency metrics (including endpointing), and generate detailed AI-powered reports.

## Features

- **One-Screen UI**: Configure and run benchmarks entirely from a user-friendly web interface.
- **Strict Batching**: Ensures accurate call tracking and respects concurrency limits.
- **Detailed Metrics**: Analyzes Vapi-reported latencies (Mean, P50, P95) and breaks them down by component (LLM, TTS, ASR, Endpointing).
- **Smart Filtering**: Automatically excludes calls that are too short (<45s) or have insufficient turns (<=2).
- **AI Reports**: Generates comprehensive markdown reports with executive summaries and detailed per-turn analysis.
- **History**: Automatically saves timestamped reports for historical tracking.

## Prerequisites

- **Python 3.8+**
- **Vapi API Key**: You need an account at [Vapi.ai](https://vapi.ai).
- **OpenRouter API Key**: Required for generating the AI summary report. Get one at [OpenRouter.ai](https://openrouter.ai).
- **Agent Configuration**: You need the Phone Numbers and IDs for:
    - **Agent A (Control)**: The baseline agent.
    - **Agent B (Test)**: The agent you are testing.
    - **Caller Agent**: An agent configured to place outbound calls to your test agents.

## Installation

1.  **Clone the repository** (or download the files):
    ```bash
    git clone <repository-url>
    cd vapi_benchmark
    ```

2.  **Create a Virtual Environment** (Recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

You can configure the tool via the UI every time you run it, but it is highly recommended to set up a `.env` file for default values.

1.  Create a `.env` file in the root directory:
    ```bash
    touch .env
    ```

2.  Add your configuration to `.env`:
    ```env
    # API Keys
    VAPI_API_KEY=your_vapi_key_here
    OPENROUTER_API_KEY=your_openrouter_key_here

    # Agent A (Control)
    AGENT_A_PHONE_NUMBER=+15550000001
    AGENT_A_PHONE_NUMBER_ID=uuid-for-agent-a

    # Agent B (Test)
    AGENT_B_PHONE_NUMBER=+15550000002
    AGENT_B_PHONE_NUMBER_ID=uuid-for-agent-b

    # Caller Agent
    AGENT_C_ASSISTANT_ID=uuid-for-caller-assistant
    AGENT_C_PHONE_NUMBER_ID=uuid-for-caller-phone

    # Defaults
    BATCH_SIZE=5
    POLL_INTERVAL_SECONDS=10
    ```

## Usage

1.  **Start the Application**:
    ```bash
    streamlit run app.py
    ```

2.  **Configure the Run**:
    - The UI will pre-fill values from your `.env` file.
    - Adjust **Calls per Agent** (e.g., 5 calls means 5 for Agent A and 5 for Agent B).
    - Adjust **Batch Size** (Default is 5. Note: 5 outbound + 5 inbound = 10 concurrent calls).

3.  **Add Context (Optional)**:
    - Use the "Report Context" text area to describe what you are testing (e.g., "Config B uses a faster LLM model"). This helps the AI generate a more meaningful report.

4.  **Run Benchmark**:
    - Click the **🚀 Run Benchmark** button.
    - Monitor the real-time logs in the UI.

5.  **View Results**:
    - Once complete, the report will be displayed automatically.
    - A box plot comparing latencies will be shown.
    - The full report is saved locally as `benchmark_report_YYYYMMDD_HHMMSS.md`.

## Troubleshooting

-   **"Command not found: streamlit"**: Ensure you activated your virtual environment and installed requirements.
-   **Concurrency Issues**: If calls are failing, try reducing the **Batch Size**.
-   **Empty Report**: Ensure your calls are longer than 45 seconds and have more than 2 turns. Short calls are filtered out to ensure data quality.

## Project Structure

-   `app.py`: The Streamlit frontend application.
-   `benchmark.py`: The core logic script that handles API calls, batching, and reporting.
-   `requirements.txt`: Python dependencies.
-   `benchmark_report_*.md`: Generated benchmark reports.
