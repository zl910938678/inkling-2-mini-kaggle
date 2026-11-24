# Inkling 2 Mini on Kaggle T4 x2

用 Kaggle 双 T4 运行 Inkling 2 Mini，并通过 Cloudflare Quick Tunnel 提供临时网页和 OpenAI 兼容 API。

## 首次配置

1. 在 Kaggle 创建或打开一个私有 Notebook，导入本仓库的 `kaggle.ipynb`。启动入口已配置为 `zl910938678/inkling-2-mini-kaggle`。
2. 启用 Internet，加速器选择 GPU T4 x2。
3. 私有仓库需要在 GitHub 创建仅限本仓库、Contents: Read-only 的 fine-grained token，并在 Kaggle Secrets 保存为 `GITHUB_READ_TOKEN`，然后授权给 Notebook。公开仓库可将 `PRIVATE_REPO` 改为 `False`。
4. 以后打开 Notebook，点击 Run All，即可拉取 `main` 分支并启动。

## 访问

启动器会验证双卡显存、执行本地推理和公网推理，然后输出 `PUBLIC_URL`、`OPENAI_BASE_URL` 和 `SESSION_KEY`。浏览器打开 `PUBLIC_URL`，输入当前会话密钥即可聊天。

API 使用 `OPENAI_BASE_URL`、模型名 `Inkling-2-Mini` 和 `SESSION_KEY`。请求使用 `stream: false`；Quick Tunnel 不支持 SSE。停止 Kaggle 会话会关闭服务，每次重启会生成新的地址和密钥。

首次运行需要下载约 18 GB 权重。只在私有 Notebook 中运行，不要公开包含 `SESSION_KEY` 的输出。
