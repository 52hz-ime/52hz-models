import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { after, before, test } from 'node:test';
import { chromium } from 'playwright';

let browser, server, url;
before(async () => {
  server = spawn(process.env.PYTHON || 'python3', ['-B', 'web_lab.py', '--port', '0'], { cwd: new URL('../../', import.meta.url) });
  const lines = createInterface({ input: server.stdout });
  for await (const line of lines) {
    url = line.match(/http:\/\/127\.0\.0\.1:\d+/)?.[0];
    if (url) break;
  }
  assert.ok(url);
  browser = await chromium.launch();
});
after(async () => { server?.kill(); await browser?.close(); });

async function pageFor(run, handler) {
  const page = await browser.newPage();
  run.after(() => page.close());
  await page.route('**/api/model', route => route.fulfill({ json: { name: 'test.gguf' } }));
  await page.route('**/api/infer', async route => {
    const request = route.request().postDataJSON();
    const result = await handler(request);
    await route.fulfill({ json: { ...result, id: request.id } });
  });
  await page.goto(url);
  return page;
}

test('More re-evaluates a locked span but does not commit; undo restores the keys', async run => {
  const requests = [];
  const page = await pageFor(run, request => {
    requests.push(request);
    if (request.event === 'more') return { action: 'ask', focus: 'kaihu', options: ['开户', '开会'] };
    return { action: request.event === 'space' ? 'commit' : 'preview', candidates: ['明天开会'] };
  });
  await page.getByLabel('实时预览').uncheck();
  const keys = page.getByLabel('原始按键');
  await keys.fill('mingtiankaihu');
  await keys.evaluate(element => element.setSelectionRange(8, 13));
  await keys.press('F2');
  await page.getByRole('button', { name: '开会', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('.preedit')?.textContent === '明天开会');
  assert.equal(requests[0].focus, 'kaihu');
  assert.equal(requests.at(-1).origin, 'more');
  assert.deepEqual(requests.at(-1).selected, { kaihu: '开会' });
  assert.equal(await keys.inputValue(), 'mingtiankaihu');
  await keys.press('Space');
  await page.waitForFunction(() => !document.querySelector('#keys').value);
  assert.equal(await page.locator('#document').innerText(), '明天开会');
  await page.getByRole('button', { name: '撤销上次上屏' }).click();
  assert.equal(await keys.inputValue(), 'mingtiankaihu');
});

test('space asks one question at a time and commits after the last answer', async run => {
  const page = await pageFor(run, request => {
    if (!request.selected.lixin) return { action: 'ask', focus: 'lixin', options: ['李欣', '李鑫'] };
    if (!request.selected.kaihu) return { action: 'ask', focus: 'kaihu', options: ['开户', '开会'] };
    return { action: 'commit', candidates: ['找李鑫开户'] };
  });
  await page.getByLabel('实时预览').uncheck();
  const keys = page.getByLabel('原始按键');
  await keys.fill('zhaolixinkaihu');
  await keys.press('Space');
  await page.getByRole('button', { name: '李鑫' }).click();
  await page.getByRole('button', { name: '开户' }).click();
  await page.waitForFunction(() => !document.querySelector('#keys').value);
  assert.equal(await page.locator('#document').innerText(), '找李鑫开户');
});

test('an old commit cannot replace newer input and native composition does not submit', async run => {
  let release, started;
  const first = new Promise(resolve => { started = resolve; });
  const gate = new Promise(resolve => { release = resolve; });
  const events = [];
  const page = await pageFor(run, async request => {
    events.push(request.event);
    started(); await gate;
    return { action: 'commit', candidates: ['旧文本'] };
  });
  await page.getByLabel('实时预览').uncheck();
  const keys = page.getByLabel('原始按键');
  await keys.fill('first');
  await keys.dispatchEvent('keydown', { key: ' ', isComposing: true });
  await keys.press('Space');
  await first;
  await keys.fill('new');
  release();
  await page.getByText('已过期', { exact: true }).waitFor();
  assert.equal(await keys.inputValue(), 'new');
  assert.equal(await page.locator('#document').innerText(), 'new');
  assert.deepEqual(events, ['space']);
});

test('punctuation-only input commits literally without calling the model', async run => {
  const calls = [];
  const page = await pageFor(run, request => { calls.push(request); return {}; });
  const keys = page.getByLabel('原始按键');
  await keys.fill('……!?「」');
  assert.equal(await keys.inputValue(), '');
  assert.equal(await page.locator('#document').innerText(), '……!?「」');
  await page.waitForTimeout(450);
  assert.deepEqual(calls, []);
  await page.getByRole('button', { name: '撤销上次上屏' }).click();
  assert.equal(await keys.inputValue(), '');
  assert.equal(await page.locator('#document').innerText(), '等待输入');
});

test('prefix and suffix bypass decoding; repeated marks and typeahead survive disambiguation', async run => {
  const requests = [];
  let started, release;
  const first = new Promise(resolve => { started = resolve; });
  const gate = new Promise(resolve => { release = resolve; });
  const page = await pageFor(run, async request => {
    requests.push(request);
    if (requests.length === 1) { started(); await gate; return { action: 'ask', focus: 'hao', options: ['好', '号'] }; }
    return { action: 'commit', candidates: [request.keys === 'nihao' ? '你好。' : '再见'] };
  });
  const keys = page.getByLabel('原始按键');
  await page.getByLabel('实时预览').uncheck();
  await keys.fill('「nihao！');
  await first;
  assert.equal(await keys.inputValue(), 'nihao！');
  await keys.press('!');
  await keys.pressSequentially('zaijian');
  await keys.press('Space');
  assert.equal(await page.locator('#document').innerText(), '「nihao！!zaijian');
  release();
  await page.getByRole('button', { name: '好', exact: true }).click();
  await page.waitForFunction(() => !document.querySelector('#keys').value);
  assert.equal(await page.locator('#document').innerText(), '「你好！!再见');
  assert.deepEqual(requests.map(({ keys, event }) => [keys, event]), [['nihao', 'space'], ['nihao', 'select'], ['zaijian', 'space']]);
});

test('ambiguous separators remain editable inside mixed input and trailing literals survive undo', async run => {
  const requests = [];
  const page = await pageFor(run, request => {
    requests.push(request.keys);
    return { action: 'commit', candidates: ['查看PR-239和v3.14'] };
  });
  await page.getByLabel('实时预览').uncheck();
  const keys = page.getByLabel('原始按键');
  await keys.pressSequentially('kanPR-239hev3.14.');
  assert.deepEqual(requests, []);
  await keys.press('Space');
  await page.waitForFunction(() => !document.querySelector('#keys').value);
  assert.deepEqual(requests, ['kanPR-239hev3.14']);
  assert.equal(await page.locator('#document').innerText(), '查看PR-239和v3.14.');
  await page.getByRole('button', { name: '撤销上次上屏' }).click();
  assert.equal(await keys.inputValue(), 'kanPR-239hev3.14.');
  assert.equal(await page.locator('#document').innerText(), '查看PR-239和v3.14.');
});
