import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { useObject } from '@ai-sdk/react';
import { z } from 'zod';
import { Activity, ArrowDownToLine, Braces, Check, ChevronDown, Clock3, Cpu, Ellipsis, Keyboard, ListFilter, LoaderCircle, Radio, RefreshCw, SlidersHorizontal, Space, Trash2, Undo2 } from 'lucide-react';
import './style.css';

const presets = {
  meeting: ['项目沟通', 'mingtiankaihu', '项目进度还没对齐，得把大家叫到一起。', ''],
  bank: ['银行预约', 'mingtiankaihu', '资料带齐了，银行那边也预约好了。', ''],
  names: ['同音联系人', 'mingtianzhaolixinkaihu', '明天的安排还没定。', '李欣是项目同事，经常组织进度会。\n李鑫是银行客户经理，负责办理账户。'],
  mixed: ['中英混输', 'qingCodexkanPR-239', '代码评审。', ''],
};
const text = z.string().min(1);
function splitKeys(raw) {
  const prefix = raw.match(/^\p{P}+/u)?.[0] || '';
  const rest = raw.slice(prefix.length);
  const suffix = rest.match(/\p{P}+$/u)?.[0] || '';
  return { prefix, body: rest.slice(0, rest.length - suffix.length), suffix };
}
const endsSentence = suffix => /[。！？，、；：…!?;,\p{Pe}\p{Pf}]/u.test(suffix);
const literal = text => ({ text, keys: '', selected: {} });
const search = z.object({ pool: z.array(text), winner: text.optional(), generate_ms: z.number(), judge_ms: z.number().optional() }).optional();
const decision = z.discriminatedUnion('action', [
  ...['preview', 'commit'].map(action => z.object({ id: text, action: z.literal(action), candidates: z.array(text).min(1).max(4), search })),
  z.object({ id: text, action: z.literal('ask'), focus: text, options: z.array(text).min(2).max(4), search }),
]);

function RequestJob({ job, finish }) {
  const { submit, stop } = useObject({ api: '/api/infer', id: job.row.id, schema: decision,
    onFinish: ({ object, error }) => finish(job, object, error?.message),
    onError: error => finish(job, null, error.message),
  });
  useEffect(() => { submit(job.row); return stop; }, []);
  return null;
}

function IconButton({ label, icon: Icon, ...props }) {
  return <button className="icon-button" title={label} aria-label={label} {...props}><Icon size={17} aria-hidden="true" /></button>;
}

const initial = { keys: '', held: '', finishRest: false, preview: '', question: null, selected: {}, entries: [], error: '',
  context: presets.meeting[2], memory: '', preset: 'meeting', live: true };

function App() {
  const [session, setSession] = useState(initial);
  const current = useRef(initial);
  const [job, setJob] = useState(null);
  const [history, setHistory] = useState([]);
  const [model, setModel] = useState('');
  const [filter, setFilter] = useState('all');
  const revision = useRef(0), serial = useRef(0), timer = useRef(), active = useRef(null), pending = useRef(null);
  const input = useRef(null), composing = useRef(false);
  const update = fields => { current.current = { ...current.current, ...fields }; setSession(current.current); };
  const documentText = session.entries.map(entry => entry.text).join('');
  const draft = splitKeys(session.held || session.keys);

  async function checkModel() {
    try {
      const response = await fetch('/api/model');
      if (!response.ok) throw Error();
      setModel((await response.json()).name);
    } catch { setModel('offline'); }
  }
  useEffect(() => { checkModel(); return () => clearTimeout(timer.current); }, []);

  function invalidate() {
    clearTimeout(timer.current);
    pending.current = null;
    revision.current++;
  }
  function start(next) { active.current = next; setJob(next); }
  function send(event, extra = {}) {
    const state = current.current;
    const { body } = splitKeys(state.held || state.keys);
    if (!body.trim()) return;
    const context = [state.context, state.entries.map(entry => entry.text).join('')].filter(Boolean).join('\n').slice(-2048);
    const row = { id: `ui-${++serial.current}`, event, keys: body, context,
      memory: state.memory.split('\n').map(line => line.trim()).filter(Boolean), selected: state.selected, ...extra };
    const next = { row, revision: revision.current, time: performance.now() };
    if (active.current) pending.current = next; else start(next);
  }
  function finish(done, result, failure) {
    if (active.current !== done) return;
    const state = current.current, row = done.row;
    const stale = done.revision !== revision.current;
    let error = failure;
    if (!stale && !error) {
      const editing = row.event === 'type' || (row.event === 'select' && row.origin === 'more');
      if (!result || result.id !== row.id || (editing && result.action !== 'preview') ||
          (row.event === 'more' && result.action !== 'ask') || (!editing && result.action === 'preview')) {
        error = '模型动作与当前输入状态不一致。';
      } else if (result.action === 'ask') {
        const options = [...new Set(result.options)];
        if (options.length < 2 || !row.keys.includes(result.focus) || (row.focus && result.focus !== row.focus) ||
            (row.event !== 'more' && Object.hasOwn(row.selected, result.focus))) error = '模型没有返回可用的局部选项。';
        else update({ question: { focus: result.focus, options, origin: row.origin || row.event }, error: '' });
      } else if (Object.values(row.selected).some(choice => !result.candidates[0].includes(choice))) {
        error = '模型改写了已选片段，已保留原输入。';
      } else {
        const raw = state.held || state.keys;
        const { suffix } = splitKeys(raw);
        const translation = suffix ? result.candidates[0].replace(/\p{P}+$/u, '') : result.candidates[0];
        if (result.action === 'preview') update({ preview: translation, question: null, error: '' });
        else {
          const rest = state.keys.slice(raw.length);
          update({ entries: [...state.entries, { keys: raw, selected: row.selected, text: translation + suffix, translation }],
            keys: '', held: '', finishRest: false, preview: '', selected: {}, question: null, error: '' });
          if (rest) changeKeys(rest, false, state.finishRest);
        }
      }
    }
    if (!stale && error) update({ error });
    const record = { row, result, error, label: stale ? '已过期' : error ? '错误' : result.action,
      duration: Math.round(performance.now() - done.time) };
    setHistory(items => [record, ...items].slice(0, 50));
    const next = pending.current;
    pending.current = null;
    start(next?.revision === revision.current ? next : null);
  }
  function schedule() {
    if (current.current.live && current.current.keys.trim() && !composing.current)
      timer.current = setTimeout(() => send('type'), 400);
  }
  function edit(fields) {
    invalidate();
    update({ preview: '', held: '', finishRest: false, question: null, error: '', ...fields });
    schedule();
  }
  function changeKeys(raw, afterComposition = false, submit = false) {
    const state = current.current;
    // A punctuation-ended segment keeps its place while the user types ahead.
    if (state.held && raw.startsWith(state.held) && !afterComposition) {
      const extra = raw.slice(state.held.length);
      update({ keys: raw, held: state.held + splitKeys(extra).prefix, finishRest: false });
      return;
    }
    const { prefix, body, suffix } = splitKeys(raw);
    const previous = splitKeys(state.held || state.keys);
    const sameBody = !afterComposition && !prefix && !!body && body === previous.body;
    if (sameBody && raw === state.keys) return;
    const held = endsSentence(suffix) ? body + suffix : '';
    const resolving = [pending.current, active.current].some(next => next?.revision === revision.current && next.row.event === 'space');
    if (sameBody && (state.question || (suffix && !state.held && (!held || resolving)))) {
      update({ keys: raw, held, finishRest: false });
      return;
    }
    invalidate();
    update({ keys: body + suffix, held, finishRest: false, entries: prefix ? [...state.entries, literal(prefix)] : state.entries,
      selected: Object.fromEntries(Object.entries(state.selected).filter(([part]) => body.includes(part))),
      preview: sameBody ? state.preview : '', question: null, error: '' });
    if (body && (held || submit)) send('space');
    else schedule();
  }
  function command(event) {
    const state = current.current;
    if (!state.keys.trim()) return;
    if (event === 'space' && state.held && state.keys.length > state.held.length &&
        [active.current, pending.current].some(next => next?.revision === revision.current &&
          (next.row.event === 'space' || next.row.origin === 'space'))) {
      update({ finishRest: true });
      return;
    }
    invalidate();
    update({ question: null, error: '' });
    const { body } = splitKeys(state.held || state.keys);
    const focus = body.slice(input.current.selectionStart, input.current.selectionEnd) || body;
    send(event, event === 'more' ? { focus } : {});
    input.current.focus();
  }
  function select(choice) {
    const question = current.current.question;
    if (!question) return;
    invalidate();
    update({ selected: { ...current.current.selected, [question.focus]: choice }, question: null, error: '' });
    send('select', { origin: question.origin });
    input.current.focus();
  }
  function undo() {
    const state = current.current, last = state.entries.at(-1);
    if (!last || state.keys) return;
    invalidate();
    update({ entries: state.entries.slice(0, -1), keys: last.keys, selected: last.selected, preview: last.translation || '', error: '', question: null });
    input.current.focus();
  }
  function keyDown(event) {
    if (event.nativeEvent.isComposing || event.nativeEvent.keyCode === 229 || composing.current) return;
    if (event.key === 'Escape') { invalidate(); update({ question: null, held: '', finishRest: false, error: '', preview: '' }); return; }
    if (event.key === 'F2') { event.preventDefault(); command('more'); return; }
    if (event.key === ' ' && !event.shiftKey && !event.ctrlKey && !event.metaKey && !event.altKey) {
      event.preventDefault();
      if (event.repeat || current.current.question) return;
      if (current.current.keys.trim()) command('space');
      else { invalidate(); update({ entries: [...current.current.entries, literal(' ')], keys: '', preview: '' }); }
    }
  }
  function loadPreset(name) {
    const [, keys, context, memory] = presets[name];
    invalidate();
    update({ preset: name, keys, context, memory, held: '', finishRest: false, selected: {}, preview: '', question: null, error: '', entries: [] });
    schedule(); input.current.focus();
  }
  const visibleHistory = history.filter(item => filter === 'all' || item.row.event === filter);
  const last = history[0];
  return <>
    {job && <RequestJob key={job.row.id} job={job} finish={finish} />}
    <header className="mast"><div className="mast-inner"><div className="brand"><img src="/whale.svg" alt="" width="38" height="38" /><strong>52Hz</strong><span className="rule"/><span>Input Lab</span></div>
      <div className="connection"><i className={model && model !== 'offline' ? 'dot online' : 'dot'} />{model === 'offline' ? '模型离线' : model ? '本地模型已连接' : '检查连接'}<IconButton icon={RefreshCw} label="刷新模型连接" onClick={checkModel}/></div>
    </div></header>
    <main><div className="title-row"><div><p className="eyebrow"><Radio size={14} aria-hidden="true"/> IME WORKBENCH</p><h1>输入法实验台</h1></div>
      <div className="model-tag"><Cpu size={16} aria-hidden="true"/><span>{model && model !== 'offline' ? model.replace(/\.gguf$/i, '') : 'llama.cpp'}</span></div></div>
    <div className="workbench"><section className="editor"><div className="section-title"><h2><Keyboard/>输入工作区</h2><IconButton icon={Undo2} label="撤销上次上屏" disabled={!session.entries.length || !!session.keys} onClick={undo}/></div>
      <div className="paper"><div className="paper-label">文本 <span>/ OUTPUT</span></div><div id="document" role="status" className="document">{documentText || (!session.keys && <span className="placeholder">等待输入</span>)}{session.keys && <>{draft.prefix}<span className="preedit">{session.preview || draft.body}</span>{draft.suffix}{session.held && session.keys.slice(session.held.length)}</>}</div></div>
      <div className="field-heading"><label htmlFor="keys">原始按键</label><span>{session.keys.length} / 256</span></div>
      <input id="keys" ref={input} maxLength={256} autoComplete="off" spellCheck="false" autoCapitalize="off" placeholder="mingtiankaihu" value={session.keys}
        onKeyDown={keyDown} onChange={event => {
          if (composing.current || event.nativeEvent.isComposing) update({ keys: event.target.value, preview: '', question: null });
          else changeKeys(event.target.value);
        }} onCompositionStart={() => { composing.current = true; invalidate(); }}
        onCompositionEnd={event => { composing.current = false; changeKeys(event.currentTarget.value, true); }}/>
      {session.question && <div className="choices"><strong>选择译法 <code>{session.question.focus}</code></strong><div>{session.question.options.map(choice => <button key={choice} onClick={() => select(choice)}>{choice}</button>)}</div></div>}
      {session.error && <div className="error" role="alert">{session.error}</div>}
      <div className="actions"><button className="primary" disabled={!session.keys.trim() || !!session.question} onClick={() => command('space')}><Space/>空格上屏</button><button className="secondary" disabled={!session.keys.trim()} title="选中按键片段后按 F2，或点击此处" onClick={() => command('more')}><Ellipsis/>More</button>
        <span className="activity">{job ? <><LoaderCircle className="spin"/>推理中</> : <><Check/>就绪</>}</span></div>
    </section>
    <aside className="settings"><div className="section-title"><h2><SlidersHorizontal/>实验设置</h2></div>
      <label htmlFor="preset">测试场景</label><div className="preset"><select id="preset" value={session.preset} onChange={event => loadPreset(event.target.value)}>{Object.entries(presets).map(([id, [name]]) => <option key={id} value={id}>{name}</option>)}</select><IconButton icon={ArrowDownToLine} label="载入此场景，重置输入区" onClick={() => loadPreset(session.preset)}/></div>
      <label htmlFor="context">上下文</label><textarea id="context" rows="2" maxLength={1600} value={session.context} onChange={event => edit({ context: event.target.value })}/>
      <label htmlFor="memory">个人记忆</label><textarea id="memory" rows="2" maxLength={1600} placeholder="每行一条，例如：项目对接人叫李欣" value={session.memory} onChange={event => edit({ memory: event.target.value })}/>
      <label className="toggle"><input type="checkbox" checked={session.live} onChange={event => edit({ live: event.target.checked })}/><span>实时预览</span><small>400 ms</small></label>
    </aside></div>
    <section className="trace"><div className="section-title"><h2><Activity/>调用记录 <small>{history.length}</small></h2><div className="trace-tools"><ListFilter size={16} aria-hidden="true"/><select aria-label="筛选事件" value={filter} onChange={event => setFilter(event.target.value)}><option value="all">全部事件</option>{['type','space','more','select'].map(kind => <option key={kind}>{kind}</option>)}</select><IconButton icon={Trash2} label="清空调用记录" disabled={!history.length} onClick={() => setHistory([])}/></div></div>
      <div className="trace-head"><span>事件 / 按键</span><span>动作</span><span>耗时</span></div>
      {!visibleHistory.length && <div className="empty"><Activity size={20} aria-hidden="true"/>{history.length ? '没有匹配的调用' : '还没有模型调用'}</div>}
      {visibleHistory.map(item => <details key={item.row.id}><summary><span className="summary-keys"><ChevronDown size={14}/><b>{item.row.event}</b><code>{item.row.keys}</code></span><span>{item.label}</span><span>{item.duration} ms</span></summary><div className="payload"><div><h3>请求</h3><pre>{JSON.stringify(item.row, null, 2)}</pre></div><div><h3>响应</h3>{item.error && <p className="error">{item.error}</p>}<pre>{JSON.stringify(item.result, null, 2)}</pre></div></div></details>)}
    </section><footer><span><Braces/> Vercel AI SDK UI</span><span><Clock3/> {last ? `${last.duration} ms` : '暂无耗时'}</span></footer></main>
  </>;
}

createRoot(document.getElementById('root')).render(<App/>);
