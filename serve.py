import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import tarfile
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path('/kaggle/tmp/inkling')
MODEL = 'Inkling-2-Mini'
TAG = 'b12600'
HF = 'https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/resolve/e8a2cecf0e78914f07cc27b9be711a9b2c7561d9'
GH = f'https://github.com/ggml-org/llama.cpp/releases/download/{TAG}'
BACKEND = 'http://127.0.0.1:8080'


def download(url, path, sha=None):
    if path.is_file():
        return
    part = path.with_suffix(path.suffix + '.part')
    print(f'Downloading {path.name}', flush=True)
    subprocess.run(['curl', '-fLsS', '--retry', '3', '--connect-timeout', '20',
                    '--speed-time', '60', '--speed-limit', '1000', '-o', str(part), url], check=True)
    if sha:
        with part.open('rb') as f:
            actual = hashlib.file_digest(f, 'sha256').hexdigest()
        if actual != sha:
            part.unlink()
            raise RuntimeError(f'Checksum mismatch for {path.name}')
    part.replace(path)


def gpu(field):
    return subprocess.check_output(['nvidia-smi', f'--query-gpu={field}',
                                    '--format=csv,noheader,nounits'], text=True).splitlines()


def request(url, payload=None, key=None, timeout=8):
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = 'Bearer ' + key
    body = None if payload is None else json.dumps(payload).encode()
    return urlopen(Request(url, body, headers), timeout=timeout)


def check_status(url):
    with request(url + '/health') as response:
        return json.load(response).get('status') == 'ok'


def wait_for(check, process, log, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and process.poll() is None:
        try:
            if result := check():
                return result
        except (URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(2)
    raise RuntimeError(log.name + ' startup failed:\n' + '\n'.join(log.read_text(errors='replace').splitlines()[-30:]))


def smoke(url, key=None):
    payload = {'model': MODEL, 'messages': [{'role': 'user', 'content': 'Name one color in English.'}],
               'stream': False, 'max_tokens': 64}
    with request(url + '/v1/chat/completions', payload, key, timeout=95) as response:
        result = json.load(response)['choices'][0]['message']
    if not (result.get('content') or result.get('reasoning_content')):
        raise RuntimeError('Inference returned no text')


def stream_smoke(url, key=None):
    payload = {'model': MODEL, 'messages': [{'role': 'user', 'content': 'Name one color in English.'}],
               'stream': True, 'max_tokens': 64}
    chunks = []
    with request(url + '/v1/chat/completions', payload, key, timeout=95) as response:
        if response.headers.get_content_type() != 'text/event-stream':
            raise RuntimeError('Streaming response did not use text/event-stream')
        for line in response:
            if line.startswith(b'data: ') and line.strip() != b'data: [DONE]':
                chunks.append(line)
    if not chunks:
        raise RuntimeError('Streaming inference returned no chunks')


UI = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Inkling 2 Mini</title><style>
*{box-sizing:border-box}body{margin:0;background:#f7f8fa;color:#172a26;font:16px system-ui}
main{width:min(100% - 32px,700px);margin:5vh auto}h1{font-size:1.8rem}label{display:block;font-weight:600;margin:1rem 0 .4rem}
input,textarea{width:100%;font:inherit;padding:.65rem;border:1px solid #819790;border-radius:6px}textarea{resize:vertical}
button{margin:.65rem .35rem 0 0;padding:.5rem .8rem;border-radius:6px;border:1px solid #206044;background:#206044;color:white;font:inherit}button+button{background:white;color:#206044}button:disabled{opacity:.65}
article{border-top:1px solid #c6d1cd;margin-top:1rem;padding-top:.7rem}p{white-space:pre-wrap;overflow-wrap:anywhere}details{color:#53655e;font-size:.9rem}summary{cursor:pointer}
</style><main><h1>Inkling 2 Mini</h1><p>双 T4 · 实时流式对话</p><form id="chat"><label for="key">会话密钥</label>
<input type="password" id="key" autocomplete="off" required><div id="history" aria-live="polite"></div>
<label for="message">消息</label><textarea id="message" rows="4" required maxlength="8000"></textarea>
<button id="send">发送</button><button type="button" id="stop" disabled>停止生成</button><button type="button" id="clear">清空对话</button><p id="status" role="status"></p></form></main>
<script>
const get = (id) => document.getElementById(id);
const key = get('key'), message = get('message'), list = get('history'),
send = get('send'), clear = get('clear'), stop = get('stop'), statusLine = get('status'), messages = [];
let controller;
function append(label, text) {
 const row = document.createElement('article'), title = document.createElement('strong'), body = document.createElement('p');
 title.textContent = label; body.textContent = text; row.append(title, body); list.append(row);
 return {row, body};
}
clear.onclick = () => { messages.length = 0; list.replaceChildren(); statusLine.textContent = ''; };
stop.onclick = () => controller?.abort();
async function readEvents(response, onEvent) {
 const reader = response.body.getReader(), decoder = new TextDecoder();
 let buffer = '';
 try {
  while (true) {
   const {value, done} = await reader.read();
   buffer += decoder.decode(value, {stream: !done});
   const frames = buffer.split(/\r?\n\r?\n/); buffer = frames.pop();
   for (const frame of frames) {
    const data = frame.split(/\r?\n/).filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).join('\n');
    if (!data) continue;
    if (data === '[DONE]') { await reader.cancel(); return; }
    const event = JSON.parse(data);
    if (event.error) throw new Error(event.error.message || String(event.error));
    onEvent(event);
   }
   if (done) throw new Error('连接提前结束，请重试。');
  }
 } finally { reader.releaseLock(); }
}
get('chat').onsubmit = async (ev) => {
 ev.preventDefault(); const content = message.value.trim(); if (!content) return;
 if (controller) return;
 controller = new AbortController();
 send.disabled = clear.disabled = key.disabled = true; stop.disabled = false; statusLine.textContent = '正在连接…';
 append('你', content);
 const answer = append('Inkling', ''), details = document.createElement('details'), summary = document.createElement('summary'), reasoningBody = document.createElement('p');
 summary.textContent = '思考过程'; details.append(summary, reasoningBody); details.hidden = true; details.open = true; answer.row.insertBefore(details, answer.body);
 const user = {role:'user',content}; let output = '', reasoning = '', finishReason = '', complete = false;
 try {
  const response = await fetch('/v1/chat/completions', {method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+key.value.trim()},
   body:JSON.stringify({model:'Inkling-2-Mini',messages:[...messages,user],stream:true,max_tokens:512}), signal:controller.signal});
  if (!response.ok) throw new Error(response.status === 401 ? '会话密钥不正确。' : response.status === 429 ? '模型正在处理其他请求，请稍后重试。' : '请求失败（HTTP '+response.status+'）。');
  if (!response.headers.get('Content-Type')?.includes('text/event-stream')) throw new Error('服务未返回流式响应。');
  await readEvents(response, event => {
   const choice = event.choices?.[0]; if (!choice) return;
   const delta = choice.delta || {};
   if (delta.reasoning_content) { reasoning += delta.reasoning_content; reasoningBody.textContent = reasoning; details.hidden = false; statusLine.textContent = '正在思考…'; }
   if (delta.content) { output += delta.content; answer.body.textContent = output; details.open = false; statusLine.textContent = '正在生成…'; }
   if (choice.finish_reason) finishReason = choice.finish_reason;
  });
  if (!output && !reasoning) throw new Error('模型没有返回文本。');
  complete = true;
  statusLine.textContent = finishReason === 'length' ? '已达到本次输出上限。' : '生成完成';
 } catch (err) { statusLine.textContent = err.name === 'AbortError' ? '已停止生成。' : err.message; }
 finally {
  if (output) { messages.push(user,{role:'assistant',content:output}); message.value = ''; }
  else if (complete && reasoning) statusLine.textContent += ' 当前仅返回了思考过程，可重试或简化问题。';
  controller = null; send.disabled = clear.disabled = key.disabled = false; stop.disabled = true; message.focus();
 }
};
</script></html>'''.encode('utf-8')


class Gateway(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *_):
        pass

    def respond(self, code, body, mime='application/json'):
        self.send_response(code)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def authorized(self):
        if secrets.compare_digest(self.server.auth, self.headers.get('Authorization', '').encode()):
            return True
        self.respond(401, b'{"error":"Valid session key required"}')
        return False

    def proxy(self, payload=None):
        try:
            with request(BACKEND + self.path, payload, timeout=100) as reply:
                self.respond(reply.status, reply.read())
        except HTTPError as err:
            self.respond(err.code, err.read())
        except (URLError, TimeoutError):
            self.respond(504, b'{"error":"Model unavailable or timed out"}')

    def proxy_stream(self, payload):
        started = False
        try:
            with request(BACKEND + self.path, payload, timeout=100) as reply:
                self.send_response(reply.status)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache, no-transform')
                self.send_header('X-Accel-Buffering', 'no')
                self.send_header('Connection', 'close')
                self.end_headers()
                started = True
                for line in reply:
                    self.wfile.write(line)
                    self.wfile.flush()
        except HTTPError as err:
            self.respond(err.code, err.read())
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (URLError, TimeoutError):
            if not started:
                self.respond(504, b'{"error":"Model unavailable or timed out"}')
            else:
                try:
                    self.wfile.write(b'data: {"error":{"message":"Model stream timed out"}}\n\n')
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

    def do_GET(self):
        if self.path == '/':
            self.respond(200, UI, 'text/html; charset=utf-8')
        elif self.path == '/health' or (self.path == '/v1/models' and self.authorized()):
            self.proxy()
        elif self.path != '/v1/models':
            self.respond(404, b'{"error":"Not found"}')

    def do_POST(self):
        if self.path != '/v1/chat/completions':
            self.respond(404, b'{"error":"Not found"}')
            return
        if not self.authorized():
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 8 * 1024**2:
                raise ValueError
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict) or type(payload.get('stream', False)) is not bool:
                raise ValueError
            limit = payload.setdefault('max_tokens', 512)
            if type(limit) is not int or not 1 <= limit <= 1024:
                raise ValueError
        except (ValueError, UnicodeError):
            self.respond(400, b'{"error":"Use a JSON object, boolean stream and max_tokens from 1 to 1024"}')
            return
        if not self.server.slot.acquire(blocking=False):
            self.respond(429, b'{"error":"Inference slot is busy. Try again shortly."}')
            return
        try:
            if payload.get('stream', False):
                self.proxy_stream(payload)
            else:
                self.proxy(payload)
        finally:
            self.server.slot.release()


def main():
    names = gpu('name')
    if len(names) != 2 or not all('T4' in name for name in names):
        raise RuntimeError(f'Expected GPU T4 x2; found {names}')
    print('Detected:', names, flush=True)
    base_vram = [int(x) for x in gpu('memory.used')]
    runtime = ROOT / TAG
    runtime.mkdir(parents=True, exist_ok=True)
    for prefix, sha in (
        ('llama', '6b2bbca74c951e03e2cb45771410ac58590b6677257ae78e2c8a896bb0a3e1b2'),
        ('cudart-llama', '971775211e47dad69f8143eab527893abaacdd69b91005ff3621c72692cfe199'),
    ):
        asset = f'{prefix}-{TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz'
        archive = ROOT / asset
        download(f'{GH}/{asset}', archive, sha)
        with tarfile.open(archive) as tar:
            tar.extractall(runtime, filter='data')
    binary = next(runtime.rglob('llama-server'))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1')
    libs = sorted({str(p.parent) for p in runtime.rglob('*.so*')})
    env['LD_LIBRARY_PATH'] = ':'.join(filter(None, [*libs, env.get('LD_LIBRARY_PATH')]))
    subprocess.run([str(binary), '--list-devices'], env=env, check=True)
    weights = ROOT / 'Inkling-2-Mini-IQ4_NL.gguf'
    download(f'{HF}/{weights.name}', weights, 'cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4')
    key = secrets.token_urlsafe(24)
    backend_log, tunnel_log = ROOT / 'llama.log', ROOT / 'tunnel.log'
    children = []
    gateway = None
    try:
        with backend_log.open('w') as log:
            llama = subprocess.Popen([str(binary), '--host', '127.0.0.1', '--port', '8080',
                                      '--model', str(weights), '--alias', MODEL, '--n-gpu-layers', '999',
                                      '--split-mode', 'layer', '--tensor-split', '1,3', '--ctx-size', '4096',
                                      '--parallel', '1', '--batch-size', '256', '--ubatch-size', '128', '--jinja'],
                                     stdout=log, stderr=subprocess.STDOUT, env=env)
        children.append(llama)
        wait_for(lambda: check_status(BACKEND), llama, backend_log, 600)
        vram = [int(x) for x in gpu('memory.used')]
        if any(used - before < 1024 for used, before in zip(vram, base_vram, strict=True)):
            raise RuntimeError(f'Model did not load on both GPUs: {base_vram} -> {vram} MiB')
        print(f'Both T4s loaded: {vram} MiB used', flush=True)
        smoke(BACKEND)
        print('Local inference passed', flush=True)
        stream_smoke(BACKEND)
        print('Local streaming inference passed', flush=True)
        gateway = ThreadingHTTPServer(('127.0.0.1', 7860), Gateway)
        gateway.auth = f'Bearer {key}'.encode()
        gateway.slot = threading.Lock()
        threading.Thread(target=gateway.serve_forever, daemon=True).start()
        with tunnel_log.open('w') as log:
            tunnel = subprocess.Popen([
                'ssh', '-T', '-p', '443',
                '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'IdentityFile=none',
                '-o', 'ConnectTimeout=20',
                '-o', 'StrictHostKeyChecking=accept-new',
                '-o', f'UserKnownHostsFile={ROOT / "known_hosts"}',
                '-o', 'ServerAliveInterval=30',
                '-o', 'ServerAliveCountMax=3',
                '-o', 'ExitOnForwardFailure=yes',
                '-R', '80:127.0.0.1:7860',
                'nokey@localhost.run',
            ], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        children.append(tunnel)

        def public_ready():
            match = re.search(r'https://[a-z0-9-]+(?:\.app)?\.lhr\.life', tunnel_log.read_text(errors='replace'))
            if not match:
                match = re.search(r'https://[a-z0-9-]+\.localhost\.run', tunnel_log.read_text(errors='replace'))
            if match and check_status(match[0]):
                return match[0]

        url = wait_for(public_ready, tunnel, tunnel_log, 180)
        smoke(url, key)
        stream_smoke(url, key)
        print(f'PUBLIC_URL={url}\nOPENAI_BASE_URL={url}/v1\nSESSION_KEY={key}', flush=True)
        while all(proc.poll() is None for proc in children):
            time.sleep(5)
        raise RuntimeError('llama-server or SSH tunnel stopped unexpectedly')
    finally:
        for proc in reversed(children):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        if gateway:
            gateway.shutdown()
            gateway.server_close()


if __name__ == '__main__':
    main()
