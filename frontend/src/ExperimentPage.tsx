import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { Archive, BookOpen, ChevronRight, CircleStop, Download, FlaskConical, Play,
  RotateCcw, Settings2 } from 'lucide-react';
import { Link, useSearchParams } from 'react-router-dom';
import type { SetURLSearchParams } from 'react-router-dom';
import { active, api, exportUrl, rowsOf } from './api';
import type { Experiment, Field, Parameters, Run, RunEvent, RunMeta, Stage, Value } from './api';
import { Notice, PageHeading } from './UI';
import { categories, lessons } from './content';
import { DataTable, Overlay, PayoffMatrices, Trace } from './Dashboard';

type Models = {available: boolean; models: string[]; message?: string};

const defaults = (exp: Experiment) => Object.fromEntries(exp.fields.map(field => [field.key, field.default]));

export function visibleFields(exp: Experiment, values: Parameters) {
  return exp.fields.filter(field => Object.entries(field.when || {}).every(([key, allowed]) => allowed.includes(values[key])));
}

export function estimateModelCalls(exp: string, p: Parameters) {
  if (exp === 'l4') return (p.model_secondary ? 2 : 1) * Number(p.prefixes) * (Number(p.warmup) + 5 * Number(p.rounds));
  return Number(p.rounds) * (exp === 'l2' ? 2 : 1);
}

function StageProgress({ events, meta }: { events: RunEvent[]; meta?: RunMeta }) {
  const latest = events.at(-1);
  const rounds = events.filter(e => e.type === 'round').length;
  const completed = Math.max(rounds, meta?.completed_rounds || 0);
  const total = meta?.total_rounds || 0;
  const calls = events.filter(e => e.type === 'round' || e.type === 'warmup').length;
  return <div className="run-progress" aria-live="polite">
    <div><span><span className={`status-dot ${meta?.status === 'failed' || meta?.status === 'interrupted' ? 'error' : meta ? 'ready' : ''}`}/>{meta?.status === 'cancelling' ? 'Stopping after current work' :
      meta ? meta.status.charAt(0).toUpperCase() + meta.status.slice(1) : 'Awaiting configuration'}</span>
      <strong>{completed} / {total || '—'} scored rounds</strong></div>
    <progress max={total || 1} value={completed} aria-label="Completed evaluation rounds"/>
    {meta?.model_calls != null && <small>{calls} / {meta.model_calls} model calls recorded</small>}
    {latest?.type === 'training' && <small>Training · {latest.data.condition} · {latest.data.step} / {latest.data.total}</small>}
    {latest?.type === 'warmup' && <small>Warm-up · {latest.data.condition} · {latest.data.step} / {latest.data.total}</small>}
  </div>;
}

function FieldControl({ field, value, change, models }: {
  field: Field; value: Value; change: (v: Value) => void; models?: Models;
}) {
  const id = `field-${field.key}`;
  const isModel = field.kind.endsWith('model');
  const describedBy = field.hint ? `${id}-hint` : undefined;
  return <div className="field">
    <label htmlFor={id}>{field.label}</label>
    {field.kind === 'select' || isModel ? <select id={id} aria-describedby={describedBy} value={value}
      required={field.kind !== 'optional_model'} onChange={event => change(event.target.value)}>
      {isModel && <option value="">{field.kind === 'optional_model' ? 'One model only' :
        models?.models.length ? 'Choose a model' : 'No model available'}</option>}
      {isModel && value && !models?.models.includes(String(value)) && <option value={value}>{value} (saved)</option>}
      {Object.entries(field.options ?? Object.fromEntries((models?.models ?? []).map(name => [name, name])))
        .map(([key, description]) => <option key={key} value={key}>{description}</option>)}
    </select> : <input id={id} aria-describedby={describedBy} value={value} onChange={event => change(
      field.kind === 'number' && event.target.value !== '' ? Number(event.target.value) : event.target.value)}
      type={field.kind} min={field.min} max={field.max} step={field.step} required={field.kind === 'number'}/>}
    {field.hint && <small id={describedBy}>{field.hint}</small>}
  </div>;
}

function Lesson({ exp }: { exp: Experiment }) {
  const lesson = lessons[exp.id];
  return <section className="educational">
    <div className="section-heading"><div><p className="eyebrow"><BookOpen size={15}/> ALGORITHM WALKTHROUGH</p><h2>How this experiment works</h2></div></div>
    <div className="panel lesson">
      <div className="lesson-overview"><h3>Objective</h3><p>{lesson.objective}</p>
        <h3>What to watch during simulation</h3><p>{lesson.observe}</p></div>
      <details open><summary>Algorithm and worked example</summary>
        <ol>{lesson.steps.map(step => <li key={step}>{step}</li>)}</ol><h4>Worked example</h4><p>{lesson.example}</p></details>
      <details><summary>Inputs and mathematical formulation</summary>
        <dl className="key-values">{exp.fields.map(field => <div key={field.key}><dt>{field.label}</dt><dd>
          {field.hint || (field.options ? Object.values(field.options).join(' · ') :
            field.kind.endsWith('model') ? 'Choose a model installed in your local Ollama library.' : `Range ${field.min}–${field.max}, default ${field.default}.`)}</dd></div>)}</dl>
        <pre className="formula">{lesson.math}</pre>
        <PayoffMatrices game={exp.game}/>
      </details>
      <details><summary>Results, limitations, and research connection</summary>
        <h4>How to read the results</h4><p>{lesson.results}</p><h4>Limitations</h4><p>{lesson.limitation}</p>
        <h4>Connection to the research</h4><p>{lesson.connection}</p></details>
    </div>
  </section>;
}

export function ExperimentPage({ exp }: { exp: Experiment }) {
  const [query, setQuery] = useSearchParams();
  const runId = query.get('run');
  const [models, setModels] = useState<Models>();
  const [checking, setChecking] = useState(false);
  const isLlm = exp.category === 'llm';

  async function refreshModels() {
    setChecking(true);
    try {
      setModels(await api<Models>('/ollama/models'));
    } catch (err) {
      setModels({available: false, models: [], message: (err as Error).message});
    } finally {
      setChecking(false);
    }
  }

  useEffect(() => { if (isLlm) void refreshModels(); }, [isLlm]);

  return <ExperimentRun key={runId} exp={exp} runId={runId} setQuery={setQuery}
    models={models} checking={checking} refreshModels={refreshModels}/>;
}

function ExperimentRun({ exp, runId, setQuery, models, checking, refreshModels }: {
  exp: Experiment; runId: string | null; setQuery: SetURLSearchParams;
  models?: Models; checking: boolean; refreshModels: () => void;
}) {
  const [configuration, setValues] = useState<Parameters>(() => defaults(exp));
  const [meta, setMeta] = useState<RunMeta>();
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [result, setResult] = useState<Run>();
  const [condition, setCondition] = useState('');
  const [error, setError] = useState('');
  const [reconnecting, setReconnecting] = useState(false);
  const [requesting, setRequesting] = useState(false);
  const [traceVersion, setTraceVersion] = useState(0);
  const mounted = useRef(false);
  const isLlm = exp.category === 'llm';
  const values = isLlm && !configuration.model && models?.models.length ?
    {...configuration, model: models.models[0]} : configuration;
  const fields = visibleFields(exp, values);
  const busy = requesting || active(meta?.status) || (!!runId && !meta && !error);
  const validation = exp.id === 's6';

  // A start response may navigate, so invalidate its handler when this run leaves the page.
  useLayoutEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  useEffect(() => {
    if (!runId) return;
    let live = true;
    let stream: EventSource | undefined;
    let timer: number | undefined;
    let pending: RunEvent[] = [];
    const flush = () => {
      window.clearTimeout(timer);
      timer = undefined;
      const batch = pending;
      pending = [];
      if (live && batch.length) setEvents(current => [...current, ...batch]);
    };
    const finish = async () => {
      try {
        const data = await api<Run>(`/simulations/${runId}/results`);
        if (!live) return;
        setMeta(data);
        setResult(data);
        setEvents(data.events);
        setReconnecting(false);
        if (data.status === 'failed') setError(data.message || 'Simulation failed.');
        if (data.status === 'interrupted') setError(data.message || 'The simulation was interrupted.');
      } catch (err) {
        if (live) setError((err as Error).message);
      }
    };
    api<RunMeta>(`/simulations/${runId}`).then(data => {
      if (!live) return;
      if (data.experiment_id !== exp.id) throw new Error('This saved run belongs to a different experiment.');
      setMeta(data);
      setValues({...defaults(exp), ...data.parameters});
      setCondition(data.conditions[0] || '');
      if (!active(data.status)) {
        void finish();
        return;
      }
      stream = new EventSource(`/api/simulations/${runId}/events`);
      stream.onopen = () => { if (live && stream) setReconnecting(false); };
      stream.onmessage = event => {
        if (!live || !stream) return;
        const next: RunEvent = JSON.parse(event.data);
        pending.push(next);
        if (timer === undefined) timer = window.setTimeout(flush, 50);
        if (next.type === 'started') setMeta(current => current?.status === 'queued' ? {...current, status: 'running'} : current);
        if (next.type === 'completed' || next.type === 'cancelled' || next.type === 'failed' || next.type === 'interrupted') {
          flush();
          stream.close();
          stream = undefined;
          setReconnecting(false);
          const status = next.type;
          setMeta(current => current && {...current, status});
          void finish();
        }
      };
      stream.onerror = () => { if (live && stream) setReconnecting(true); };
    }).catch(err => { if (live) setError((err as Error).message); });
    return () => { live = false; window.clearTimeout(timer); stream?.close(); };
  }, [runId, exp]);

  async function start(event: FormEvent) {
    event.preventDefault();
    setRequesting(true);
    setError('');
    try {
      const parameters = Object.fromEntries(fields.map(field => [field.key, values[field.key]]));
      const data = await api<RunMeta>('/simulations', {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({experiment_id: exp.id, parameters})});
      if (mounted.current) setQuery({run: data.id}, {replace: true});
    } catch (err) {
      if (mounted.current) {
        setError((err as Error).message);
        setRequesting(false);
      }
    }
  }

  async function stop() {
    if (!meta || !active(meta.status)) return;
    try {
      const data = await api<RunMeta>(`/simulations/${meta.id}/cancel`, {method: 'POST'});
      if (mounted.current) setMeta(current => current && active(current.status) ? data : current);
    } catch (err) {
      if (mounted.current) setError((err as Error).message);
    }
  }

  function reset() {
    setQuery({}, {replace: true});
    setError('');
    setValues(defaults(exp));
    setTraceVersion(value => value + 1);
  }

  const rows = useMemo(() => rowsOf(events), [events]);
  const conditionRows = rows.filter(row => row.condition === condition);
  const training = events.flatMap(event => event.type === 'training' &&
    (exp.id === 's5' || event.data.condition === condition) ? [event.data as Stage] : []);
  const availableConditions = meta?.conditions || [];
  const grouping = (name: string) => name.split(' · ').slice(0, -1).join(' · ');
  const comparable = validation || exp.id === 'l4' ?
    availableConditions.filter(name => grouping(name) === grouping(condition)) : availableConditions;
  const traces = comparable.map(name => ({name, rows: rows.filter(row => row.condition === name)}))
    .filter(trace => trace.rows.length);
  const lesson = lessons[exp.id];
  const category = categories[exp.category];
  return <>
    <Link className="back-link" to={`/category/${exp.category}`}>{category.title}</Link>
    <PageHeading eyebrow="RUN SIMULATION" title={exp.title} description={exp.description}>
      {(isLlm || exp.category === 'neural' || validation) && <Link className="button secondary" to={`/replay/${isLlm ? 'ollama-controlled' : 'neural-validation'}`}><Archive size={17}/> Replay Research Results</Link>}
    </PageHeading>
    <div className="architecture" aria-label="Algorithm architecture">{lesson.architecture.map((node, index) => <div key={node}>
      {!!index && <ChevronRight size={16}/>}<span>{node}</span></div>)}</div>
    {error && <Notice danger>{error}</Notice>}
    {reconnecting && <Notice>Reconnecting to the event stream. The simulation may still be running.</Notice>}
    <div className="simulation-layout">
      <div className="simulation-controls">
        <form className="panel config" onSubmit={start}>
          <div className="panel-heading"><h2><Settings2 size={18}/> Configuration</h2><FlaskConical size={17}/></div>
          {isLlm && models?.available === false && <Notice>{models.message} No models are downloaded by the lab.</Notice>}
          {isLlm && models?.available && !models.models.length && <Notice>Ollama is running with an empty model library.</Notice>}
          <fieldset className="config-fields" disabled={busy}>
            {fields.map(field => <FieldControl key={field.key} field={field} value={values[field.key]} models={models}
              change={value => setValues(current => ({...current, [field.key]: value}))}/>)}
          </fieldset>
          {validation && <div className="model-estimate"><span>
            <strong>{Number(values.seed_count) * 2 * (values.strategy === 'all' ? 3 : 1)}</strong> planned neural episodes
          </span></div>}
          {isLlm && <div className="model-estimate"><span><strong>{estimateModelCalls(exp.id, values)}</strong> planned model calls</span>
            <button type="button" className="text-button" disabled={checking || busy} onClick={refreshModels}>{checking ? 'Checking…' : 'Refresh models'}</button></div>}
          {exp.id === 'r3' && values.strategy === 'hedge' && <Notice>Both opponents use Hedge. Their moves may diverge in response to the two learners.</Notice>}
          <div className="config-actions">
            <button className="button primary" type="submit" disabled={busy || (isLlm && (!models?.available || !models.models.includes(String(values.model)) || checking))}><Play size={17}/>
              {requesting ? 'Starting…' : 'Run Simulation'}</button>
            <button className="button secondary" type="button" disabled={!busy || !meta || meta.status === 'cancelling'} onClick={stop}><CircleStop size={17}/> Stop</button>
            <button className="button secondary" type="button" disabled={busy} onClick={reset}><RotateCcw size={16}/> Reset</button>
          </div>
        </form>
        {result && <div className="panel export-panel"><h3>Download run</h3><div>
          <a className="button secondary compact" href={exportUrl(result.id, 'csv')}><Download size={15}/> CSV</a>
          <a className="button secondary compact" href={exportUrl(result.id, 'json')}><Download size={15}/> JSON</a>
        </div></div>}
      </div>
      <div className="simulation-main">
        <StageProgress events={events} meta={meta}/>
        {!!meta && meta.conditions.length > 1 && <label className="inline-field condition-picker">Condition
          <select value={condition} onChange={event => setCondition(event.target.value)}>{meta.conditions.map(name => <option key={name}>{name}</option>)}</select>
        </label>}
        <Trace key={`${condition}-${traceVersion}`} rows={conditionRows} game={exp.game} training={training}
          caption={exp.game === 'bait' ? 'General-sum game' : 'Zero-sum RPS'} activity={active(meta?.status)}
          summary={result?.summary.find(item => item.condition === condition)}/>
        {busy && <p className="playback-note">Playback controls affect the visualization. Stop interrupts execution at the next work boundary.</p>}
      </div>
    </div>
    {traces.length > 1 && <section className="comparison-section"><div className="section-heading"><h2>Compare conditions</h2></div>
      <Overlay traces={traces}/>{result && !validation && <DataTable title="Final metrics" rows={result.summary}/>}</section>}
    {exp.id === 'l4' && !!result?.paired?.length && <DataTable title="Paired differences versus Always Scissors" rows={result.paired}/>}
    {validation && !!result?.paired?.length && <DataTable title="Paired changes (λ=1 minus λ=0) · 95% seed bootstrap intervals" rows={result.paired}/>}
    {validation && result && <DataTable title="Per-seed evaluation metrics" rows={result.summary}/>}
    <Lesson exp={exp}/>
  </>;
}
