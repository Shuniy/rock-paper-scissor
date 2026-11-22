import { useEffect, useState } from 'react';
import { Archive, BrainCircuit, ChevronRight, Download, GitCompareArrows, MessageSquare, Play } from 'lucide-react';
import { Link, useParams } from 'react-router-dom';
import { active, api, download, downloadCsv, rowsOf } from './api';
import type { Dataset, Experiment, Historical, Run, RunMeta, TabularRow } from './api';
import { Notice, PageHeading } from './UI';
import { Chart, DataTable, label, Overlay, Trace } from './Dashboard';

const strategies: Record<string, string> = {uniform: 'Uniform', myopic: 'Myopic', plan2: 'Planner-2',
  always_scissors: 'Always Scissors', bait2_scissors: 'Bait 2', bait4_scissors: 'Bait 4', adaptive: 'Adaptive'};

function mean(values: number[]) {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

export function ArchivePage() {
  const {dataset} = useParams();
  const [index, setIndex] = useState<Dataset[]>([]);
  const [recorded, setRecorded] = useState<Historical>();
  const [runs, setRuns] = useState<RunMeta[]>([]);
  const [error, setError] = useState('');
  const [runsError, setRunsError] = useState('');
  useEffect(() => {
    let live = true;
    api<Dataset[]>('/historical-results').then(data => {if (live) setIndex(data);})
      .catch(err => {if (live) setError(err.message);});
    api<RunMeta[]>('/simulations').then(data => {if (live) setRuns(data);})
      .catch(err => {if (live) setRunsError(err.message);});
    return () => { live = false; };
  }, []);
  useEffect(() => {
    let live = true;
    setRecorded(undefined);
    setError('');
    if (dataset) api<Historical>(`/historical-results/${dataset}`).then(data => {if (live) setRecorded(data);})
      .catch(err => {if (live) setError(err.message);});
    return () => { live = false; };
  }, [dataset]);

  const selected = index.find(item => item.id === dataset);
  return <>
    <PageHeading eyebrow="REPLAY MODE" title={selected?.title || 'Replay Research Results'}
      description={selected?.description || 'Explore the original study data and replay your saved runs without training a model or calling Ollama.'}>
      <Link className="button secondary" to={dataset ? '/replay' : '/experiment/r1'}>{dataset ? <Archive size={17}/> : <Play size={17}/>}
        {dataset ? 'Back to archive' : 'Run Simulation'}</Link>
    </PageHeading>
    {error && <Notice danger>{error}</Notice>}
    {!dataset && <>
      <section className="archive-section"><div className="section-heading"><h2>Previously recorded research experiments</h2></div>
        <div className="catalog-grid archive-grid">{index.map(item => {
          const Icon = item.category === 'neural' ? BrainCircuit : MessageSquare;
          return <Link key={item.id} to={`/replay/${item.id}`} className="catalog-card">
            <div className="archive-card-head"><span className={`mode-icon ${item.category === 'neural' ? 'lavender' : 'sky'}`}><Icon size={21}/></span>
              <span className="pill">{item.category === 'neural' ? 'Per-seed summaries' : 'Round-level replay'}</span></div>
            <h2>{item.title}</h2><p>{item.description}</p><span className="text-link">Open recorded results <ChevronRight size={17}/></span>
          </Link>;
        })}</div>
        {!index.length && !error && <p role="status">Loading the research archive…</p>}
      </section>
      <section className="archive-section"><div className="section-heading"><h2>Your local simulations</h2></div>
        {runsError && <Notice danger>{runsError}</Notice>}
        {runs.length ? <div className="panel saved-runs">{runs.map(run => <Link key={run.id} to={`/experiment/${run.experiment_id}?run=${run.id}`}>
          <span><strong>{run.experiment_id.toUpperCase()} · {run.conditions.join(' / ')}</strong>
            <small>{new Date(run.created_at).toLocaleString()} · {run.completed_rounds} scored rounds · {run.status}</small></span>
          <span className="text-link">{active(run.status) ? 'Follow run' : 'Open run'} <ChevronRight size={17}/></span>
        </Link>)}</div> : <div className="panel empty-state"><Archive size={24}/><p>Saved simulations will appear here after the first run.</p></div>}
      </section>
    </>}
    {dataset && !recorded && !error && <p role="status">Loading the recorded data…</p>}
    {recorded && <section className="archive-detail">
      <div className="archive-source"><div><strong>Previously recorded research experiments</strong>
        <p>{recorded.aggregate_only ? 'The source contains one summary per seed, strategy, and objective.' :
          'The source contains evaluation rounds and episode summaries. Warm-up histories were outside these files.'}</p></div>
        <div className="export-buttons">
          {selected?.files.map((file, i) => <a key={file} className="button secondary compact"
            href={`/api/historical-results/${dataset}/files/${i}`}><Download size={15}/>{file.split('/').at(-1)}</a>)}
          <button type="button" className="button secondary compact" onClick={() => download(`${dataset}.json`, JSON.stringify(recorded, null, 2), 'application/json')}><Download size={15}/> JSON</button>
        </div></div>
      {recorded.aggregate_only ? <NeuralArchive key={dataset} data={recorded}/> : <OllamaArchive key={dataset} data={recorded}/>}
    </section>}
  </>;
}

function NeuralArchive({ data }: { data: Historical }) {
  const [strategy, setStrategy] = useState('myopic');
  const [metric, setMetric] = useState('expected_b');
  const options = [...new Set(data.rows.map(row => String(row.strategy)))];
  const rows = data.rows.filter(row => row.strategy === strategy);
  const seeds = [...new Set(rows.map(row => Number(row.seed)))].sort((a, b) => a - b);
  const points = seeds.map(seed => ({seed, reward: rows.find(row => row.seed === seed && row.lambda_swap === 0)?.[metric],
    swap: rows.find(row => row.seed === seed && row.lambda_swap === 1)?.[metric]}));
  return <>
    <Link className="button primary" to="/experiment/s6"><Play size={17}/> Run paired validation</Link>
    <Notice>Per-seed aggregates are available for this study. The chart uses seed on the horizontal axis.</Notice>
    <div className="filter-bar">
      <label className="inline-field">Strategy <select value={strategy} onChange={event => setStrategy(event.target.value)}>
        {options.map(option => <option value={option} key={option}>{strategies[option] || option}</option>)}</select></label>
      <label className="inline-field">Metric <select value={metric} onChange={event => setMetric(event.target.value)}>
        {['expected_b', 'realized_b', 'ext_a', 'swap_a', 'a_entropy'].map(key => <option value={key} key={key}>{label(key)}</option>)}</select></label>
    </div>
    <Chart title={`Matched initialization seeds · ${label(metric)}`} data={points} x="seed" xLabel="Initialization seed"
      series={[{key: 'reward', name: 'Reward only (λ=0)'}, {key: 'swap', name: 'Reward + swap (λ=1)'}]}/>
    <DataTable title="Paired changes (λ=1 minus λ=0) · 95% seed bootstrap intervals" rows={data.paired || []}/>
    <DataTable title="Recorded seed summaries" rows={rows}/>
    <p className="caption">Each pair matches its initialization seed and A’s sampling draws. Both conditions can have different opponent trajectories. Intervals resample seed pairs.</p>
  </>;
}

function OllamaArchive({ data }: { data: Historical }) {
  const [selected, setSelected] = useState(data.runs?.[0]?.condition || '');
  const [metric, setMetric] = useState('b_reward');
  const models = [...new Set(data.rows.map(row => String(row.model)))];
  const options = [...new Set(data.rows.map(row => String(row.strategy)))];
  const points = options.map(strategy => ({
    strategy: strategies[strategy] || strategy,
    ...Object.fromEntries(models.map((model, i) => [`m${i}`,
      mean(data.rows.filter(row => row.model === model && row.strategy === strategy).map(row => Number(row[metric])))])),
  }));
  const activeRoundTrace = data.runs?.find(run => run.condition === selected);
  const episode = data.rows.find(row => row.model === activeRoundTrace?.model && row.prefix === activeRoundTrace?.prefix && row.strategy === activeRoundTrace?.strategy);
  const phaseMean = (phase: string) => {
    const rows = activeRoundTrace?.rounds.filter(row => row.phase === phase) || [];
    return rows.length ? mean(rows.map(row => row.reward_b)) : null;
  };
  return <>
    <Link className="button primary" to="/experiment/l4"><Play size={17}/> Run a controlled comparison</Link>
    <div className="filter-bar">
      <label className="inline-field">Study metric <select value={metric} onChange={event => setMetric(event.target.value)}>
        {['b_reward', 'paper_post', 'entropy', 'invalid'].map(key => <option key={key} value={key}>{label(key)}</option>)}</select></label>
    </div>
    <Chart title={`Recorded strategy means · ${label(metric)}`} data={points} x="strategy" series={models.map((name, i) => ({key: `m${i}`, name}))}/>
    <p className="caption">Strategies reused prefixes within each model. The recorded models had their own prefixes and two episodes per strategy.</p>
    <div className="section-heading"><h2>Replay an Ollama episode</h2></div>
    <label className="inline-field episode-select">Recorded episode <select value={selected} onChange={event => setSelected(event.target.value)}>
      {data.runs?.map(run => <option key={run.condition} value={run.condition}>{run.model} · Prefix {run.prefix} · {strategies[run.strategy] || run.strategy}</option>)}</select></label>
    {activeRoundTrace && <Trace key={selected} rows={activeRoundTrace.rounds} game="rps" caption="Recorded episode" startPaused
      summary={episode && {...episode, bait_b: phaseMean('bait'), exploit_b: phaseMean('exploit')}}/>}
    <DataTable title="Paired differences versus Always Scissors" rows={data.paired || []}/>
    <DataTable title="Recorded episode summaries" rows={data.rows}/>
  </>;
}

const protocol: Record<string, string> = {
  r1: 'Regret learners', r2: 'Regret learners', r3: 'Regret learners', n1: 'Synthetic prediction',
  n2: 'Synthetic prediction', n3: 'Synthetic reward policy', n4: 'Synthetic reward episodes',
  s1: 'Constructed bait game', s2: 'Constructed bait game', s3: 'Online Transformer RPS', s4: 'Online Transformer RPS',
};

export function incompatibility(selected: RunMeta[]) {
  if (selected.length < 2) return '';
  const [first, ...rest] = selected;
  if (rest.some(run => run.game !== first.game)) return 'The runs use different games or reward scales. Choose runs from the same game.';
  if (rest.some(run => run.parameters.rounds !== first.parameters.rounds)) return 'The runs have different evaluation lengths. Choose the same number of scored rounds.';
  if (rest.some(run => (protocol[run.experiment_id] || run.experiment_id) !== (protocol[first.experiment_id] || first.experiment_id)))
    return 'The runs use different evaluation protocols or datasets.';
  return '';
}

function configurationDifferences(selected: RunMeta[], experiments: Experiment[]) {
  const [first, ...rest] = selected;
  const keys = new Set(selected.flatMap(run => Object.keys(run.parameters)));
  const fields = experiments.filter(exp => selected.some(run => run.experiment_id === exp.id)).flatMap(exp => exp.fields);
  return [...keys].filter(key => rest.some(run => run.parameters[key] !== first.parameters[key]))
    .map(key => (fields.find(field => field.key === key)?.label || key.replaceAll('_', ' ')).toLowerCase());
}

export function ComparePage({ experiments }: { experiments: Experiment[] }) {
  const [runs, setRuns] = useState<RunMeta[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [loaded, setLoaded] = useState<Run[]>([]);
  const [conditions, setConditions] = useState<Record<string, string>>({});
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [detailsLoading, setDetailsLoading] = useState(false);
  useEffect(() => {
    let live = true;
    api<RunMeta[]>('/simulations').then(value => {if (live) setRuns(value.filter(run => run.status === 'completed'));})
      .catch(err => {if (live) setError(err.message);}).finally(() => {if (live) setLoading(false);});
    return () => { live = false; };
  }, []);
  const chosen = runs.filter(run => selected.includes(run.id));
  const invalid = incompatibility(chosen);
  const differences = configurationDifferences(chosen, experiments);
  useEffect(() => {
    let live = true;
    setLoaded([]);
    if (selected.length >= 2 && !invalid) {
      setDetailsLoading(true);
      Promise.all(selected.map(id => api<Run>(`/simulations/${id}/results`))).then(value => {
        if (live) setLoaded(value);
      }).catch(err => {if (live) setError(err.message);}).finally(() => {if (live) setDetailsLoading(false);});
    } else setDetailsLoading(false);
    return () => { live = false; };
  }, [selected, invalid]);

  const title = (run: RunMeta) => experiments.find(exp => exp.id === run.experiment_id)?.title || run.experiment_id.toUpperCase();
  const traces = loaded.map((run, i) => {
    const name = conditions[run.id] || run.conditions[0];
    return {name: `${i + 1}. ${name}`, rows: rowsOf(run.events).filter(row => row.condition === name), id: run.id,
      summary: run.summary.find(row => row.condition === name), parameters: run.parameters};
  });
  const summaries = traces.map(trace => ({run: trace.name, ...trace.summary}));
  const configurationRows: TabularRow[] = traces.map(trace => ({run: trace.name, ...trace.parameters}));
  const exportRows = traces.flatMap(trace => trace.rows.map(row => ({
    run: trace.name, run_id: trace.id, round: row.round, cumulative_a: row.cumulative_a, cumulative_b: row.cumulative_b,
    average_a: row.average_a, average_b: row.average_b, external: row.external, swap: row.swap,
  })));
  return <>
    <PageHeading eyebrow="RUN ANALYSIS" title="Experiment Comparison" description="Overlay compatible runs and inspect their final metrics and configurations.">
      <Link className="button secondary" to="/replay"><Archive size={17}/> Research archive</Link>
    </PageHeading>
    {error && <Notice danger>{error}</Notice>}
    <section className="panel data-panel compare-picker">
      <div className="panel-heading"><h2><GitCompareArrows size={19}/> Select completed runs</h2><span className="pill">{chosen.length} selected</span></div>
      {loading && <p role="status">Loading saved runs…</p>}
      {!loading && runs.length < 2 && <p>Run at least two compatible simulations to compare their curves. <Link className="text-link" to="/experiment/r1">Open the simulator</Link></p>}
      {runs.map(run => <label className="compare-option" key={run.id}><input type="checkbox"
        checked={selected.includes(run.id)} onChange={event => setSelected(current => event.target.checked ?
          [...current, run.id] : current.filter(id => id !== run.id))}/>
        <span><strong>{title(run)}</strong><small>{new Date(run.created_at).toLocaleString()} · {run.parameters.rounds} rounds per condition · seed {run.parameters.seed}</small></span>
      </label>)}
    </section>
    {invalid && <Notice>{invalid}</Notice>}
    {detailsLoading && <p role="status">Loading comparable run results…</p>}
    {traces.length >= 2 && <section className="compare-results">
      <div className="archive-source"><p>{differences.length ?
        `Descriptive comparison. Run settings differ for ${differences.join(', ')}.` :
        'Matched run parameters. Adaptive trajectories can still differ.'}</p>
        <div className="export-buttons"><button className="button secondary compact" onClick={() => downloadCsv('comparison.csv', exportRows)}><Download size={15}/> CSV</button>
          <button className="button secondary compact" onClick={() => download('comparison.json', JSON.stringify(traces, null, 2), 'application/json')}><Download size={15}/> JSON</button></div></div>
      <div className="filter-bar">{loaded.map((run, i) => <label key={run.id} className="inline-field">Run {i + 1} condition
        <select value={conditions[run.id] || run.conditions[0]} onChange={event => setConditions(current => ({...current, [run.id]: event.target.value}))}>
          {run.conditions.map(condition => <option key={condition}>{condition}</option>)}</select></label>)}</div>
      <Overlay traces={traces}/>
      <DataTable title="Final metrics" rows={summaries}/>
      <DataTable title="Configurations" rows={configurationRows}/>
    </section>}
  </>;
}
