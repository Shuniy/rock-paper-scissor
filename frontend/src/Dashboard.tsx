import { useEffect, useState } from 'react';
import { Activity, ChartNoAxesCombined, Circle, FileText, Gem, Hand, Pause, Play, RotateCcw, Scissors,
  SkipForward, Swords, Zap } from 'lucide-react';
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import type { Round, Stage, TabularRow } from './api';
import { actionLabels, baitActions } from './content';

export const metricLabels: Record<string, string> = {
  average_a: 'A reward / round', average_b: 'B reward / round', cumulative_a: 'A total reward',
  cumulative_b: 'B total reward', external: 'External regret', swap: 'Swap regret', ext_a: 'A external regret / round',
  swap_a: 'A swap regret / round', realized_b: 'B realized reward / round', b_reward: 'B reward / round',
  external_per_round: 'External regret / round', swap_per_round: 'Swap regret / round',
  expected_a: 'Expected A reward', expected_b: 'Expected B reward', predicted_return: 'B plan return',
  prediction_accuracy: 'Prediction accuracy', prediction_tv: 'Prediction error (TV)', policy_entropy: 'Policy entropy',
  swap_surrogate: 'Expected swap surrogate', update_swap_surrogate: 'Update swap surrogate', training_loss: 'Learning loss', loss: 'Training loss',
  b_reward_delta: 'Δ B reward / round', paper_post_delta: 'Δ Paper frequency',
  paper_post: 'Paper frequency from round 5', a_entropy: 'A policy entropy', entropy: 'Action entropy', invalid: 'Invalid responses',
  action_entropy: 'Action entropy', bait_b: 'B bait reward / round', exploit_b: 'B exploit reward / round',
  dynamic_regret: 'Dynamic regret', update_expected_reward: 'Update expected reward', lambda_swap: 'Swap coefficient λ',
  expected_b_delta: 'Δ B expected reward', swap_a_delta: 'Δ A swap regret', a_entropy_delta: 'Δ A entropy',
  expected_b_low: 'B reward CI low', expected_b_high: 'B reward CI high', swap_a_low: 'Swap regret CI low', swap_a_high: 'Swap regret CI high',
  a_entropy_low: 'Entropy CI low', a_entropy_high: 'Entropy CI high',
};

export function label(key: string) {
  return metricLabels[key] || key.replaceAll('_', ' ').replace(/^./, char => char.toUpperCase());
}
export function format(value: unknown, digits = 3): string {
  if (typeof value === 'number') return Number(value.toFixed(digits)).toString();
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  return value == null ? '—' : String(value);
}

const colors = ['var(--mint)', 'var(--amber)', 'var(--lavender)', 'var(--sky)', 'var(--rose)', 'var(--text)'];
const actionIcons = { Rock: Gem, Paper: FileText, Scissors, X: Zap, Y: Circle, Bait: Hand, Exploit: Swords, Safe: Circle };

function Move({ action, reward, agent }: { action?: string; reward?: number; agent: 'a' | 'b' }) {
  const Icon = (action && actionIcons[action as keyof typeof actionIcons]) || Circle;
  return <div className={`agent agent-${agent}`}>
    <span className="agent-label">AGENT {agent.toUpperCase()} <small>{agent === 'a' ? 'Learner' : 'Opponent'}</small></span>
    <div className="move-icon"><Icon size={34}/></div>
    <strong>{action || 'Waiting'}</strong><small className="payoff">{reward == null ? '—' : `${reward > 0 ? '+' : ''}${format(reward)}`} payoff</small>
  </div>;
}

function Arena({ row, caption, game }: { row?: Round; caption: string; game: string }) {
  return <div className="arena">
    <div className="panel-heading"><div><p className="eyebrow">SIMULATION ARENA</p><h2>{row ? `Round ${row.round}` : 'Ready to play'}</h2></div><span className="pill">{caption}</span></div>
    <div className="duel"><Move action={row?.a} reward={row?.reward_a} agent="a"/><span className="versus">VS</span>
      <Move action={row?.b} reward={row?.reward_b} agent="b"/></div>
    {row && <>
      <div className="arena-scores"><span>A total <b>{format(row.cumulative_a)}</b></span>
        <span>B total <b>{format(row.cumulative_b)}</b></span><span>External regret <b>{format(row.external)}</b></span>
        <span>Swap regret <b>{format(row.swap)}</b></span><span>Phase <b>{row.phase || 'Evaluation'}</b></span></div>
      {row.policy && <div className="arena-policy" aria-label="Agent A policy">
        {(game === 'bait' ? baitActions : actionLabels).map((action, i) => <span key={action}>
          <span>{action}</span><b>{format(100 * row.policy![i], 1)}%</b>
        </span>)}
      </div>}
    </>}
  </div>;
}

export function Chart({ title, data, x, series, xLabel }: {
  title: string; data: Record<string, unknown>[]; x: string; series: { key: string; name: string }[]; xLabel?: string;
}) {
  if (!data.length || !series.length) return null;
  return <div className="panel chart" role="region" aria-label={title}>
    <h3>{title}</h3>
    <div className="chart-frame">
      <ResponsiveContainer width="100%" height="100%" minWidth={0} initialDimension={{ width: 450, height: 235 }}>
        <LineChart data={data} margin={{left: -14, right: 15, top: 8, bottom: 8}}>
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 4" vertical={false}/>
          <XAxis dataKey={x} tick={{fill: 'var(--muted)', fontSize: 12}} tickLine={false} axisLine={false}
            label={xLabel ? {value: xLabel, position: 'insideBottom', offset: -5, fill: 'var(--muted)', fontSize: 11} : undefined}/>
          <YAxis tick={{fill: 'var(--muted)', fontSize: 12}} tickLine={false} axisLine={false}/>
          <Tooltip contentStyle={{background: 'var(--panel)', border: '1px solid var(--border)', color: 'var(--text)', borderRadius: 8}}
            labelStyle={{color: 'var(--text)'}} formatter={value => format(value)}/>
          <Legend wrapperStyle={{fontSize: 12, paddingTop: 14}}/>
          {series.map(({key, name}, index) => <Line type="linear" dataKey={key} name={name} key={key} stroke={colors[index % colors.length]}
            strokeWidth={2} isAnimationActive={false} dot={data.length < 15 ? {r: 2.7} : false} connectNulls={false}/>)}
        </LineChart>
      </ResponsiveContainer>
    </div>
  </div>;
}

export function RoundCharts({ rows, game, training = [] }: { rows: Round[]; game: string; training?: Stage[] }) {
  const actions = game === 'bait' ? baitActions : actionLabels;
  const keys = ['expected_a', 'expected_b', 'predicted_return', 'prediction_accuracy', 'prediction_tv', 'policy_entropy',
    'swap_surrogate', 'update_swap_surrogate', 'training_loss', 'dynamic_regret'];
  const extras = keys.filter(key => rows.some(r => typeof r.detail[key] === 'number'));
  const chartRows = rows.map(r => ({...r, ...Object.fromEntries(extras.map(key => [key, r.detail[key]])),
    ...Object.fromEntries(actions.flatMap((_, i) => [
      [`p${i}`, r.policy?.[i]], [`f${i}`, r.frequency[i]],
      [`q${i}`, Array.isArray(r.detail.prediction) ? r.detail.prediction[i] : undefined],
    ]))}));
  const series = (...keys: string[]) => keys.map(key => ({key, name: label(key)}));
  const stages = new Map<string, Stage[]>();
  for (const event of training) {
    const stage = stages.get(event.condition);
    if (stage) stage.push(event);
    else stages.set(event.condition, [event]);
  }
  const extraSeries = extras.filter(key => key !== 'dynamic_regret');
  return <div className="chart-grid">
    <Chart title="Cumulative reward" data={chartRows} x="round" series={series('cumulative_a', 'cumulative_b')}/>
    <Chart title="Average reward" data={chartRows} x="round" series={series('average_a', 'average_b')}/>
    <Chart title="Cumulative regret" data={chartRows} x="round" series={series('external', 'swap', ...extras.filter(key => key === 'dynamic_regret'))}/>
    {rows.some(row => row.policy) && <Chart title="Action probabilities" data={chartRows} x="round"
      series={actions.map((name, i) => ({name, key: `p${i}`}))}/>}
    <Chart title="Observed action frequency" data={chartRows} x="round" series={actions.map((name, i) => ({name, key: `f${i}`}))}/>
    {rows.some(row => Array.isArray(row.detail.prediction)) && <Chart title={rows.some(row => row.detail.predicted_action) ? 'Predicted B actions' : 'Predicted A actions'}
      data={chartRows} x="round" series={actions.map((name, i) => ({name, key: `q${i}`}))}/>}
    {extraSeries.length > 0 && <Chart title="Experiment metrics" data={chartRows} x="round" series={series(...extraSeries)}/>}
    {[...stages].map(([name, steps]) => {
      const metrics = ['loss', 'expected_a', 'swap_surrogate'].filter(key => steps.some(row => Object.hasOwn(row, key)));
      return <Chart key={name} title={`Training progress · ${name}`} x="step" xLabel="Iteration"
        data={metrics.length ? steps : steps.map(row => ({...row, percent: 100 * row.step / row.total}))}
        series={metrics.length ? series(...metrics) : [{key: 'percent', name: 'Completion (%)'}]}/>;
    })}
  </div>;
}

export function DataTable({ title, rows, fields, limit = 10 }: {
  title: string; rows: TabularRow[]; fields?: string[]; limit?: number;
}) {
  const [all, setAll] = useState(false);
  if (!rows.length) return null;
  const keys = fields ?? [...new Set(rows.flatMap(Object.keys))];
  return <section className="panel data-panel">
    <div className="panel-heading"><h3>{title}</h3><span className="pill">{rows.length} rows</span></div>
    <div className="table-scroll"><table><thead><tr>{keys.map(key => <th key={key} scope="col">{label(key)}</th>)}</tr></thead>
      <tbody>{rows.slice(0, all ? undefined : limit).map((row, i) => <tr key={i}>{keys.map(key => <td key={key}>{format(row[key])}</td>)}</tr>)}</tbody>
    </table></div>
    {rows.length > limit && <button type="button" className="text-button show-more" onClick={() => setAll(!all)}>{all ? 'Show fewer rows' : `Show all ${rows.length} rows`}</button>}
  </section>;
}

function ProbabilityBars({ values, actions, title }: { values: number[]; actions: string[]; title: string }) {
  return <div className="probabilities"><h4>{title}</h4>
    {values.map((value, i) => <div className="prob-row" key={actions[i]}><span>{actions[i]}</span>
      <div className="prob-track"><div style={{width: `${100 * value}%`, background: colors[i]}}/></div>
      <strong>{(100 * value).toFixed(1)}%</strong></div>)}
  </div>;
}

function Matrix({ matrix, rows, columns, title, heat = false }: {
  matrix: number[][]; rows: string[]; columns: string[]; title: string; heat?: boolean;
}) {
  return <div className="matrix-wrap"><h4>{title}</h4><table className="matrix"><thead><tr>
    <th aria-label="Row labels"/>{columns.map(name => <th key={name} scope="col">{name}</th>)}</tr></thead>
    <tbody>{matrix.map((row, i) => <tr key={rows[i]}><th scope="row">{rows[i]}</th>
      {row.map((value, j) => <td key={columns[j]} style={heat ? {background: `color-mix(in srgb, var(--mint) ${100 * (0.08 + 0.4 * value)}%, var(--panel))`} : undefined}>{format(value)}</td>)}
    </tr>)}</tbody></table></div>;
}

export function PayoffMatrices({ game }: { game: string }) {
  return <div className="matrix-grid">
    {game === 'bait' ? <>
      <Matrix title="A’s payoff" matrix={[[1, -1, 0], [0, 1, 0]]} rows={baitActions} columns={['Bait', 'Exploit', 'Safe']}/>
      <Matrix title="B’s payoff" matrix={[[-.1, 3, .2], [-.1, -3, .2]]} rows={baitActions} columns={['Bait', 'Exploit', 'Safe']}/>
    </> : <Matrix title="A’s RPS payoff (B earns the negative)" matrix={[[0, -1, 1], [1, 0, -1], [-1, 1, 0]]} rows={actionLabels} columns={actionLabels}/>}
  </div>;
}

function Inspector({ row, game }: { row: Round; game: string }) {
  const actions = game === 'bait' ? baitActions : actionLabels;
  const handled = new Set(['explanation', 'raw', 'invalid', 'fallback', 'prediction', 'next_policy', 'next_policy_frozen', 'transition', 'public_history']);
  const entries = Object.entries(row.detail).filter(([key, value]) => !handled.has(key) && value != null);
  return <section className="panel inspector">
    <div className="panel-heading"><h3>Round {row.round} inspector</h3><span className="pill">A {row.a} · B {row.b}</span></div>
    <p>{typeof row.detail.explanation === 'string' ? row.detail.explanation : 'This round is reconstructed from the recorded actions and payoff.'}</p>
    <p className="reward-explanation">A’s move selects the row and B’s move selects the column. A earned <strong>{format(row.reward_a)}</strong> and B earned <strong>{format(row.reward_b)}</strong>.</p>
    <div className="matrix-grid">
      {row.policy && <ProbabilityBars values={row.policy} actions={actions} title="A’s pre-action distribution"/>}
      {Array.isArray(row.detail.next_policy) && <ProbabilityBars values={row.detail.next_policy} actions={actions} title="A’s next distribution"/>}
      {Array.isArray(row.detail.next_policy_frozen) && <ProbabilityBars values={row.detail.next_policy_frozen}
        actions={actions} title="Next history before the gradient update"/>}
      {Array.isArray(row.detail.prediction) && <ProbabilityBars values={row.detail.prediction} actions={actions}
        title={row.detail.predicted_action ? 'Prediction of B’s move' : 'B’s estimate of A’s policy'}/>}
      {Array.isArray(row.detail.transition) && <Matrix title="Transition matrix (source × replacement)" matrix={row.detail.transition}
        rows={actions} columns={actions} heat/>}
      {Array.isArray(row.detail.pre_log_weights) && Array.isArray(row.detail.pre_log_weights[0]) &&
        <Matrix title="Row log-weights before learning" matrix={row.detail.pre_log_weights} rows={actions} columns={actions}/>}
    </div>
    {row.history !== undefined && <div className="inspection-block"><h4>History available before this decision</h4>
      {row.history.length ? <div className="history-list">{row.history.map((item, i) => <span key={i}>{typeof item === 'string' ? item :
        Object.entries(item as Record<string, unknown>).map(([key, value]) => `${key === 'a' || key === 'b' ? key.toUpperCase() : label(key)} ${value}`).join(' · ')}</span>)}</div>
        : <p>No previous rounds were shown.</p>}</div>}
    {Array.isArray(row.detail.public_history) && <details className="detail-disclosure">
      <summary>Public history available to B ({row.detail.public_history.length} completed rounds)</summary>
      <p>{row.detail.public_history.length ? row.detail.public_history.map(pair => `A ${pair.a} / B ${pair.b}`).join(' · ') : 'No completed action pairs.'}</p></details>}
    {typeof row.detail.raw === 'string' && <div className="inspection-block"><h4>Raw Ollama response</h4><pre>{row.detail.raw || '(empty response)'}</pre>
      {row.detail.invalid === true && <p className="invalid">Invalid response. Scored with the Rock fallback.</p>}</div>}
    {!!entries.length && <details className="detail-disclosure"><summary>Algorithm state and metrics</summary><dl className="key-values">
      {entries.map(([key, value]) => <div key={key}><dt>{label(key)}</dt><dd>{Array.isArray(value) ? JSON.stringify(value, (_, x) =>
        typeof x === 'number' ? Number(x.toFixed(6)) : x) : format(value)}</dd></div>)}</dl></details>}
  </section>;
}

function PhaseTimeline({ rows, selected, seek }: { rows: Round[]; selected?: number; seek: (n: number) => void }) {
  if (!rows.some(row => row.phase && ['bait', 'exploit', 'safe'].includes(row.phase))) return null;
  return <div className="phase-box"><h3>Opponent phase timeline</h3><div className="timeline">
    {rows.map(row => <button type="button" className={`phase ${row.phase} ${row.round === selected ? 'selected' : ''}`}
      key={row.round} title={`Round ${row.round} · ${row.phase}`} aria-label={`Inspect round ${row.round}, ${row.phase}`}
      onClick={() => seek(row.round)}>{row.round}</button>)}
  </div><div className="phase-legend"><span className="phase bait">Bait</span><span className="phase exploit">Exploit</span><span className="phase safe">Safe</span></div></div>;
}

export function Trace({ rows, game, caption, training = [], summary, activity, startPaused = false }: {
  rows: Round[]; game: string; caption: string; training?: Stage[]; summary?: TabularRow; activity?: boolean;
  startPaused?: boolean;
}) {
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(!startPaused);
  const [speed, setSpeed] = useState(1);
  useEffect(() => {
    if (!playing || cursor >= rows.length) return;
    const timer = window.setTimeout(() => setCursor(index => index + 1), 430 / speed);
    return () => window.clearTimeout(timer);
  }, [playing, cursor, rows.length, speed]);
  const available = rows.slice(0, cursor);
  const selected = available.at(-1);
  const seek = (n: number) => { setPlaying(false); setCursor(n); };
  return <div className="trace">
    <Arena row={selected} caption={caption} game={game}/>
    <div className="playback">
      <button type="button" className="button secondary compact" disabled={!rows.length}
        onClick={() => {if (cursor >= rows.length && !activity) {setCursor(0);setPlaying(true);} else setPlaying(!playing);}}>
        {playing && (activity || cursor < rows.length) ? <Pause size={17}/> : <Play size={17}/>}
        {playing && (activity || cursor < rows.length) ? 'Pause playback' : cursor === rows.length ? 'Replay' : 'Resume playback'}
      </button>
      <button type="button" className="icon-button" aria-label="Step to next round" disabled={cursor >= rows.length} onClick={() => seek(cursor + 1)}><SkipForward size={17}/></button>
      <button type="button" className="icon-button" aria-label="Reset playback" disabled={!rows.length} onClick={() => seek(0)}><RotateCcw size={17}/></button>
      <input type="range" aria-label="Playback position" min={0} max={Math.max(rows.length, 1)} value={cursor} disabled={!rows.length}
        onChange={event => seek(Number(event.target.value))}/>
      <span className="playback-count">{cursor} / {rows.length}</span>
      <select aria-label="Playback speed" value={speed} onChange={event => setSpeed(Number(event.target.value))}>
        {[.5, 1, 2, 5, 10].map(s => <option value={s} key={s}>{s}×</option>)}
      </select>
    </div>
    <PhaseTimeline rows={available} selected={selected?.round} seek={seek}/>
    {(!available.length && !training.length) && <div className="empty-charts"><ChartNoAxesCombined size={23}/>
      <span>{rows.length ? 'Play or step through the recorded rounds to reveal the charts.' : 'Play an experiment to see reward, regret, and action charts.'}</span></div>}
    <RoundCharts rows={available} game={game} training={training}/>
    {summary && cursor >= rows.length && rows.length > 0 && <DataTable title="Condition summary" rows={[summary]}/>}
    {selected && <Inspector row={selected} game={game}/>}
    {!!rows.length && <section className="panel data-panel">
      <div className="panel-heading"><h3><Activity size={17}/> Round event log</h3><span className="pill">{rows.length} rounds</span></div>
      <div className="table-scroll round-table"><table><thead><tr>{['Round', 'Agent A', 'Agent B', 'Payoff A', 'Payoff B', 'Phase'].map(name => <th key={name} scope="col">{name}</th>)}</tr></thead>
        <tbody>{[...rows].reverse().map(row => <tr key={row.round} className={row.round === selected?.round ? 'chosen-row' : ''}>
          <td><button type="button" className="text-button" onClick={() => seek(row.round)} aria-label={`Inspect round ${row.round}`}>#{row.round}</button></td>
          <td>{row.a}{row.detail.invalid === true && <small className="invalid"> · invalid</small>}</td><td>{row.b}</td>
          <td>{format(row.reward_a)}</td><td>{format(row.reward_b)}</td><td>{row.phase || '—'}</td>
        </tr>)}</tbody></table></div>
    </section>}
  </div>;
}

export function comparisonData(traces: {name: string; rows: Round[]}[], metric: keyof Round) {
  const points = new Map<number, Record<string, unknown>>();
  traces.forEach(({rows}, i) => rows.forEach(row => {
    const point = points.get(row.round) ?? {round: row.round};
    point[`run${i}`] = row[metric];
    points.set(row.round, point);
  }));
  return [...points.values()].sort((a, b) => Number(a.round) - Number(b.round));
}

export function Overlay({ traces }: { traces: {name: string; rows: Round[]}[] }) {
  const [metric, setMetric] = useState<keyof Round>('cumulative_a');
  const series = traces.map(({name}, i) => ({key: `run${i}`, name}));
  return <div className="overlay-chart">
    <label className="inline-field">Comparison metric <select value={metric} onChange={event => setMetric(event.target.value as keyof Round)}>
      {(['cumulative_a', 'cumulative_b', 'average_a', 'average_b', 'external', 'swap'] as const).map(key => <option key={key} value={key}>{label(key)}</option>)}</select></label>
    <Chart title={`Condition comparison · ${label(metric)}`} x="round" data={comparisonData(traces, metric)} series={series}/>
  </div>;
}
