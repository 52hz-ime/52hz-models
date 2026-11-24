"""Loopback-only UI for the same inference adapter used by make eval."""

import argparse
import json
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from mimetypes import guess_type
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

from infer import decision_contract, predict


DIST = Path(__file__).resolve().parent / 'web' / 'dist'


def validate(row):
    if not isinstance(row, dict) or not isinstance(row.get('keys'), str) or not 0 < len(row['keys']) <= 256:
        raise ValueError('按键长度应为 1 至 256 字符。')
    for key, limit in [('id', 80), ('context', 2048), ('focus', 256)]:
        value = row.get(key, '')
        if not isinstance(value, str) or len(value) > limit:
            raise ValueError(f'无效字段：{key}')
    if not row.get('id') or not row.get('event'):
        raise ValueError('缺少请求 ID 或事件。')
    decision_contract(row)
    memory = row.get('memory', [])
    if not isinstance(memory, list) or len(memory) > 32 or not all(isinstance(item, str) and len(item) <= 512 for item in memory):
        raise ValueError('记忆最多 32 条，每条不超过 512 字符。')
    selected = row.get('selected', {})
    if not isinstance(selected, dict) or not all(key and key in row['keys'] and isinstance(value, str) and 0 < len(value) <= 256 for key, value in selected.items()):
        raise ValueError('已选片段无效。')
    if row['event'] == 'more' and not (row.get('focus') and row['focus'] in row['keys']):
        raise ValueError('修改片段不在原始按键中。')
    return row


class LabHandler(BaseHTTPRequestHandler):
    def send_data(self, status, body, content_type):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def text(self, status, message):
        self.send_data(status, message.encode(), 'text/plain; charset=utf-8')

    def local(self):
        port = self.server.server_port
        hosts = {f'localhost:{port}', f'127.0.0.1:{port}'}
        if self.headers.get('Host') not in hosts or self.headers.get('Origin') not in {None, *(f'http://{host}' for host in hosts)}:
            self.text(403, '仅接受本机同源请求。')
            return False
        return True

    def do_GET(self):
        if not self.local():
            return
        path = urlsplit(self.path).path
        if path == '/api/model':
            try:
                with urlopen(self.server.endpoint + '/props', timeout=3) as upstream:
                    props = json.load(upstream)
                name = Path(props.get('model_path') or props.get('model_alias') or 'Qwen').name
            except (OSError, ValueError, HTTPException):
                self.text(503, '模型未连接，请运行 make serve。')
                return
            self.send_data(200, json.dumps({'name': name}).encode(), 'application/json')
            return
        target = (DIST / ('index.html' if path == '/' else path.lstrip('/'))).resolve()
        if DIST not in target.parents or not target.is_file():
            self.text(404, 'Not found')
            return
        self.send_data(200, target.read_bytes(), guess_type(target.name)[0] or 'application/octet-stream')

    def do_POST(self):
        if not self.local():
            return
        if self.path != '/api/infer':
            self.text(404, 'Not found')
            return
        if self.headers.get_content_type() != 'application/json':
            self.text(415, 'Expected application/json')
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 32768:
                self.text(413, 'Invalid request size')
                return
            row = validate(json.loads(self.rfile.read(length)))
        except (KeyError, ValueError, TypeError) as error:
            self.text(400, str(error))
            return
        try:
            result = predict(self.server.endpoint, row)
            if 'error' in result:
                raise ValueError(result['error'])
        except (KeyError, IndexError, TypeError, ValueError, OSError, HTTPException) as error:
            self.text(502, f'推理失败，请检查 make status。{error}')
            return
        self.text(200, json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8090)
    parser.add_argument('--model-port', type=int, default=8089)
    args = parser.parse_args()
    if not (DIST / 'index.html').is_file():
        parser.exit(1, '请先运行 make web-build。\n')
    with ThreadingHTTPServer(('127.0.0.1', args.port), LabHandler) as server:
        server.endpoint = f'http://127.0.0.1:{args.model_port}'
        print(f'52Hz 输入法实验台：http://127.0.0.1:{server.server_port}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
