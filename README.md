# Inkling 2 Mini on Kaggle T4 x2

Run Inkling 2 Mini on two Kaggle T4 GPUs and expose a temporary web page and OpenAI-compatible API through an anonymous `localhost.run` SSH tunnel. The web chat and API support real Server-Sent Events (SSE) token streaming.

## One-click run

1. Open the private [Kaggle Notebook](https://www.kaggle.com/code/earthtop/inkling-2-mini-dual-t4/edit) and click **Run All**. Its launch cell anonymously clones the latest `main` branch from this public repository. No GitHub login, token, or Kaggle Secret is needed.
2. To create another notebook, import [`kaggle.ipynb`](kaggle.ipynb), enable Internet, select **GPU T4 x2**, and click **Run All**.
3. The first run downloads about 18 GB of model weights. Wait for the log lines `PUBLIC_URL`, `OPENAI_BASE_URL`, and `SESSION_KEY`.

## Access

Open `PUBLIC_URL` in a browser and enter the `SESSION_KEY` from the current run. The launcher checks both T4 GPUs, then performs local and public checks for regular and streaming inference before printing the access values.

API client settings:

- Base URL: the logged `OPENAI_BASE_URL`, ending in `/v1`.
- API key: the current run's `SESSION_KEY`.
- Model: `Inkling-2-Mini`.
- Request settings: `stream: true` for SSE token streaming, or `stream: false` for a regular JSON response; `max_tokens` from 1 to 2048. The gateway allows up to 10 minutes without an upstream byte before reporting a timeout.

The address is temporary and may change when the tunnel reconnects. Stopping the Kaggle session or reaching its runtime limit closes the service and invalidates the session key. Run the notebook again to create a new address and key.

The repository is public only so Kaggle can fetch source anonymously. Keep the deployment notebook private and do not publish logs containing `SESSION_KEY`. No credentials belong in this repository.
