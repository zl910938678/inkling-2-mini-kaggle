# Inkling 2 Mini on Kaggle T4 x2

Run Inkling 2 Mini on two Kaggle T4 GPUs and expose a temporary web page and OpenAI-compatible API through a Cloudflare Quick Tunnel.

## One-click run

1. Open the private [Kaggle Notebook](https://www.kaggle.com/code/earthtop/inkling-2-mini-dual-t4/edit) and click **Run All**. Its launch cell anonymously clones the latest `main` branch from this public repository. No GitHub login, token, or Kaggle Secret is needed.
2. To create another notebook, import [`kaggle.ipynb`](kaggle.ipynb), enable Internet, select **GPU T4 x2**, and click **Run All**.
3. The first run downloads about 18 GB of model weights. Wait for the log lines `PUBLIC_URL`, `OPENAI_BASE_URL`, and `SESSION_KEY`.

## Access

Open `PUBLIC_URL` in a browser and enter the `SESSION_KEY` from the current run. The launcher checks both T4 GPUs and performs local and public inference smoke tests before printing the access values.

API client settings:

- Base URL: the logged `OPENAI_BASE_URL`, ending in `/v1`.
- API key: the current run's `SESSION_KEY`.
- Model: `Inkling-2-Mini`.
- Request settings: `stream: false` and `max_tokens` from 1 to 1024. Quick Tunnels do not support SSE.

Stopping the Kaggle session or reaching its runtime limit closes the service and invalidates the temporary address. Run the notebook again to create a new address and session key.

The repository is public only so Kaggle can fetch source anonymously. Keep the deployment notebook private and do not publish logs containing `SESSION_KEY`. No credentials belong in this repository.
