// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter, useNavigate } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Dataset, Experiment, Historical, Round, Run, RunEvent, RunMeta } from './api';
import { comparisonData } from './Dashboard';
import { estimateModelCalls, ExperimentPage, visibleFields } from './ExperimentPage';
import { ArchivePage, ComparePage, incompatibility } from './ResultsPages';
import App from './App';

const experiment: Experiment = {id: 'r1', category: 'regret', game: 'rps', title: 'External Regret / Hedge',
  description: 'Test the repeated game.', fields: [
    {key: 'rounds', kind: 'number', label: 'Evaluation rounds', default: 5, min: 1, max: 100},
    {key: 'strategy', kind: 'select', label: 'Opponent strategy', default: 'uniform', options: {uniform: 'Uniform', fixed: 'Fixed'}},
    {key: 'fixed_action', kind: 'select', label: 'Fixed action', default: 'Paper', options: {Paper: 'Paper', Rock: 'Rock'}, when: {strategy: ['fixed']}},
  ]};
const round: Round = {run_id: 'run-id', experiment_id: 'r1', condition: 'Hedge', round: 1, a: 'Scissors', b: 'Paper',
  reward_a: 1, reward_b: -1, cumulative_a: 1, cumulative_b: -1, average_a: 1, average_b: -1,
  external: 0, swap: 0, frequency: [0, 0, 1], policy: [.2, .4, .4], history: [],
  detail: {explanation: 'Computed from the test event.'}};
const run: RunMeta = {id: 'run-id', experiment_id: 'r1', game: 'rps', parameters: {rounds: 5, strategy: 'fixed', fixed_action: 'Paper'},
  status: 'running', created_at: '2026-11-21T00:00:00Z', completed_rounds: 0, total_rounds: 5, model_calls: null, conditions: ['Hedge']};

class TestEventSource {
  static instances: TestEventSource[] = [];
  onmessage?: (event: {data: string}) => void;
  onerror?: () => void;
  close = vi.fn();

  constructor() { TestEventSource.instances.push(this); }
  send(event: RunEvent) { this.onmessage?.({data: JSON.stringify(event)}); }
}

function json(value: unknown, status = 200) {
  return new Response(JSON.stringify(value), {status, headers: {'Content-Type': 'application/json'}});
}

function showExperiment(exp = experiment) {
  return render(<MemoryRouter><ExperimentPage exp={exp}/></MemoryRouter>);
}

const anotherRun: RunMeta = {...run, id: 'another-run', parameters: {...run.parameters, rounds: 8}, total_rounds: 8};

function showRun(initial = `?run=${run.id}`, exp = experiment) {
  function SwitchRun() {
    const navigate = useNavigate();
    return <button onClick={() => navigate(`?run=${anotherRun.id}`)}>Open another run</button>;
  }
  return render(<MemoryRouter initialEntries={[`/${initial}`]}><SwitchRun/><ExperimentPage exp={exp}/></MemoryRouter>);
}

beforeEach(() => {
  TestEventSource.instances = [];
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue(new DOMRect(0, 0, 450, 235));
  vi.stubGlobal('EventSource', TestEventSource);
  vi.stubGlobal('ResizeObserver', class {observe(){} unobserve(){} disconnect(){}});
});
afterEach(() => { cleanup(); vi.useRealTimers(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

describe('configuration', () => {
  it('shows conditional fields and estimates all model calls in a controlled run', () => {
    expect(visibleFields(experiment, {strategy: 'uniform'}).map(field => field.key)).not.toContain('fixed_action');
    expect(visibleFields(experiment, {strategy: 'fixed'}).map(field => field.key)).toContain('fixed_action');
    expect(estimateModelCalls('l2', {rounds: 7})).toBe(14);
    expect(estimateModelCalls('l4', {prefixes: 2, rounds: 12, warmup: 3, model_secondary: 'test'})).toBe(252);
  });

  it('shows an offline Ollama explanation and keeps research replay available', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => json({available: false, models: [], message: 'Start the Ollama app.'})));
    showExperiment({...experiment, id: 'l1', category: 'llm', fields: [
      {key: 'model', kind: 'model', label: 'Installed model', default: ''},
      {key: 'rounds', kind: 'number', label: 'Evaluation rounds', default: 12},
    ]});
    expect(await screen.findByText(/Start the Ollama app/)).toBeTruthy();
    expect((screen.getByRole('button', {name: 'Run Simulation'}) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByRole('link', {name: 'Replay Research Results'}).getAttribute('href')).toBe('/replay/ollama-controlled');
  });
});

describe('live runner', () => {
  it('submits parameters and plays real stream events while stop requests cancellation', async () => {
    const payloads: unknown[] = [];
    const final: Run = {...run, status: 'cancelled', completed_rounds: 1,
      summary: [{condition: 'Hedge', rounds: 1}], events: [{type: 'round', data: round}, {type: 'cancelled', data: {}}]};
    const fetcher = vi.fn(async (input: string, init?: RequestInit) => {
      if (input.endsWith('/cancel')) return json({...run, status: 'cancelling'}, 202);
      if (input.endsWith('/results')) return json(final);
      if (init?.method === 'POST') {
        payloads.push(JSON.parse(String(init.body)));
        return json(run, 202);
      }
      return json(run);
    });
    vi.stubGlobal('fetch', fetcher);
    showExperiment();
    fireEvent.change(screen.getByLabelText('Opponent strategy'), {target: {value: 'fixed'}});
    expect(screen.getByLabelText('Fixed action')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', {name: 'Run Simulation'}));
    await screen.findByText('Running', {exact: true});
    expect(payloads).toEqual([{experiment_id: 'r1', parameters: {rounds: 5, strategy: 'fixed', fixed_action: 'Paper'}}]);
    await waitFor(() => expect(TestEventSource.instances).toHaveLength(1));
    const source = TestEventSource.instances.at(-1);
    expect(source).toBeDefined();
    act(() => source!.send({type: 'round', data: round}));
    await screen.findByText('1 / 5 scored rounds');
    fireEvent.click(screen.getByRole('button', {name: 'Step to next round'}));
    expect(screen.getByRole('heading', {name: 'Round 1 inspector'})).toBeTruthy();
    expect(within(screen.getByRole('region', {name: 'Cumulative reward'})).getByText('A total reward')).toBeTruthy();
    const roundLog = screen.getByRole('heading', {name: 'Round event log'}).closest('section')!;
    expect(within(roundLog).getByText('Scissors')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', {name: 'Stop'}));
    await screen.findByText('Stopping after current work');
    expect(fetcher).toHaveBeenCalledWith('/api/simulations/run-id/cancel', {method: 'POST'});
    act(() => source!.send({type: 'cancelled', data: {}}));
    await screen.findByText('Cancelled', {exact: true});
    expect(source!.close).toHaveBeenCalled();
    expect(screen.getByRole('link', {name: 'CSV'}).getAttribute('href')).toContain('/export?format=csv');
  });

  it('displays a failed run request and leaves the controls usable', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => json({detail: 'A simulation is already running.'}, 409)));
    showExperiment();
    const button = screen.getByRole('button', {name: 'Run Simulation'}) as HTMLButtonElement;
    fireEvent.click(button);
    expect((await screen.findByRole('alert')).textContent).toContain('already running');
    expect(button.disabled).toBe(false);
  });

  it('clears progress, reconnection, and pending events when a different run opens', async () => {
    vi.stubGlobal('fetch', vi.fn(async (path: string) => json(path.endsWith(anotherRun.id) ? anotherRun : run)));
    showRun();
    await screen.findByText('Running', {exact: true});
    const oldStream = TestEventSource.instances[0];
    act(() => oldStream.send({type: 'round', data: round}));
    await screen.findByText('1 / 5 scored rounds');
    act(() => oldStream.onerror?.());
    expect(screen.getByText(/Reconnecting to the event stream/)).toBeTruthy();

    act(() => oldStream.send({type: 'round', data: {...round, round: 2}}));
    fireEvent.click(screen.getByRole('button', {name: 'Open another run'}));
    await screen.findByText('0 / 8 scored rounds');
    expect(oldStream.close).toHaveBeenCalledTimes(1);
    expect(screen.queryByText(/Reconnecting to the event stream/)).toBeNull();
    expect(screen.queryByRole('heading', {name: 'Round event log'})).toBeNull();
    expect(TestEventSource.instances).toHaveLength(2);

    act(() => {
      oldStream.send({type: 'round', data: {...round, round: 3}});
      TestEventSource.instances[1].send({type: 'round', data: {...round, run_id: anotherRun.id, a: 'Rock'}});
    });
    await screen.findByText('1 / 8 scored rounds');
    const log = screen.getByRole('heading', {name: 'Round event log'}).closest('section')!;
    expect(within(log).getByText('Rock')).toBeTruthy();
    expect(within(log).queryByText('Scissors')).toBeNull();
  });

  it('ignores late results and stop responses from the previous run', async () => {
    let deliverResult!: (response: Response) => void;
    let deliverStop!: (response: Response) => void;
    const resultResponse = new Promise<Response>(resolve => { deliverResult = resolve; });
    const stopResponse = new Promise<Response>(resolve => { deliverStop = resolve; });
    vi.stubGlobal('fetch', vi.fn(async (path: string) => {
      if (path.endsWith('/results')) return resultResponse;
      if (path.endsWith('/cancel')) return stopResponse;
      return json(path.endsWith(anotherRun.id) ? anotherRun : run);
    }));
    showRun();
    await screen.findByText('Running', {exact: true});
    fireEvent.click(screen.getByRole('button', {name: 'Stop'}));
    act(() => TestEventSource.instances[0].send({type: 'failed', data: {}}));
    fireEvent.click(screen.getByRole('button', {name: 'Open another run'}));
    await screen.findByText('0 / 8 scored rounds');
    await act(async () => {
      deliverResult(json({...run, status: 'failed', message: 'Failure from old run', summary: [], events: [{type: 'round', data: round}]}));
      deliverStop(json({...run, status: 'cancelling'}));
    });
    expect(screen.getByText('Running', {exact: true})).toBeTruthy();
    expect(screen.getByText('0 / 8 scored rounds')).toBeTruthy();
    expect(screen.queryByText('Failure from old run')).toBeNull();
    expect(screen.queryByRole('link', {name: 'CSV'})).toBeNull();
  });

  it('ignores metadata that arrives after opening another run', async () => {
    let deliverMetadata!: (response: Response) => void;
    const oldMetadata = new Promise<Response>(resolve => { deliverMetadata = resolve; });
    vi.stubGlobal('fetch', vi.fn(async (path: string) => path.endsWith(anotherRun.id) ? json(anotherRun) : oldMetadata));
    showRun();
    fireEvent.click(screen.getByRole('button', {name: 'Open another run'}));
    await screen.findByText('0 / 8 scored rounds');
    await act(async () => { deliverMetadata(json(run)); });
    expect(screen.getByText('0 / 8 scored rounds')).toBeTruthy();
    expect(TestEventSource.instances).toHaveLength(1);
  });

  it('keeps a reopened completed run selected when an earlier start responds', async () => {
    let deliverStart!: (response: Response) => void;
    const startResponse = new Promise<Response>(resolve => { deliverStart = resolve; });
    const completed: Run = {...anotherRun, status: 'completed', completed_rounds: 1,
      events: [{type: 'round', data: {...round, run_id: anotherRun.id}}], summary: [{condition: 'Hedge', rounds: 1}]};
    const fetcher = vi.fn(async (path: string, init?: RequestInit) => {
      if (init?.method === 'POST') return startResponse;
      return json(path.endsWith('/results') ? completed : {...anotherRun, status: 'completed'});
    });
    vi.stubGlobal('fetch', fetcher);
    showRun('');
    fireEvent.click(screen.getByRole('button', {name: 'Run Simulation'}));
    fireEvent.click(screen.getByRole('button', {name: 'Open another run'}));
    await screen.findByRole('link', {name: 'CSV'});
    await act(async () => { deliverStart(json(run, 202)); });
    expect(screen.getByRole('link', {name: 'CSV'}).getAttribute('href')).toContain(anotherRun.id);
    expect(fetcher.mock.calls.some(([path]) => path.endsWith(`/${run.id}`))).toBe(false);
    expect(TestEventSource.instances).toHaveLength(0);
    fireEvent.click(screen.getByRole('button', {name: 'Step to next round'}));
    expect(screen.getByRole('heading', {name: 'Round 1 inspector'})).toBeTruthy();
  });

  it('shows both GRU training stages before any evaluation rounds are scored', async () => {
    const gru = {...run, experiment_id: 's5', conditions: ['GRU predictor']};
    vi.stubGlobal('fetch', vi.fn(async () => json(gru)));
    showRun(undefined, {...experiment, id: 's5', category: 'strategy'});
    await screen.findByText('Running', {exact: true});
    const source = TestEventSource.instances[0];
    act(() => source.send({type: 'training', data: {condition: 'GRU trajectories', step: 1, total: 12}}));
    expect(await screen.findByRole('region', {name: /Training progress · GRU trajectories/})).toBeTruthy();
    act(() => source.send({type: 'training', data: {condition: 'GRU fit', step: 1, total: 4, loss: 1.2}}));
    const fit = await screen.findByRole('region', {name: /Training progress · GRU fit/});
    expect(within(fit).getByText('Training loss')).toBeTruthy();
    expect(screen.getByRole('region', {name: /Training progress · GRU trajectories/})).toBeTruthy();
    expect(screen.getByText('0 / 5 scored rounds')).toBeTruthy();
  });

  it('reopens an interrupted run with its recovery message and downloads', async () => {
    const interrupted: Run = {...run, status: 'interrupted', completed_rounds: 1, message: 'Saved progress is available.',
      summary: [{condition: 'Hedge', rounds: 1}], events: [{type: 'round', data: round}, {type: 'interrupted', data: {}}]};
    vi.stubGlobal('fetch', vi.fn(async () => json(interrupted)));
    showRun();
    expect(await screen.findByText('Saved progress is available.')).toBeTruthy();
    expect(screen.getByText('Interrupted', {exact: true})).toBeTruthy();
    expect(screen.getByText('1 / 5 scored rounds')).toBeTruthy();
    expect(screen.getByRole('link', {name: 'JSON'}).getAttribute('href')).toContain('/export?format=json');
  });
});

it('updates connectivity on health loss and recovery while retaining the catalog', async () => {
  vi.useFakeTimers();
  vi.spyOn(window, 'scrollTo').mockImplementation(() => {});
  vi.stubGlobal('localStorage', {getItem: () => null, setItem: () => {}});
  let available = true;
  const fetcher = vi.fn(async (path: string) => {
    if (path.endsWith('/experiments')) return json([experiment]);
    if (!available) throw new Error('Backend offline');
    return json({status: 'ok'});
  });
  vi.stubGlobal('fetch', fetcher);
  const view = render(<MemoryRouter initialEntries={['/category/regret']}><App/></MemoryRouter>);
  await act(() => vi.advanceTimersByTimeAsync(0));
  expect(screen.getByText('Local lab online')).toBeTruthy();
  expect(screen.getByText(experiment.title)).toBeTruthy();
  available = false;
  await act(() => vi.advanceTimersByTimeAsync(15000));
  expect(screen.getByText('API unavailable')).toBeTruthy();
  expect(screen.getByText(experiment.title)).toBeTruthy();
  available = true;
  await act(() => vi.advanceTimersByTimeAsync(15000));
  expect(screen.getByText('Local lab online')).toBeTruthy();
  view.unmount();
  await vi.advanceTimersByTimeAsync(30000);
  expect(fetcher.mock.calls.map(([path]) => path)).toEqual(['/api/experiments', '/api/health', '/api/health', '/api/health']);
});

describe('results', () => {
  it('maps the recorded values to each comparison series and rejects incompatible runs', () => {
    expect(comparisonData([{name: 'A', rows: [round]}, {name: 'B', rows: [{...round, cumulative_a: 4}]}], 'cumulative_a'))
      .toEqual([{round: 1, run0: 1, run1: 4}]);
    expect(incompatibility([run, {...run, game: 'bait'}])).toContain('different games');
    expect(incompatibility([run, {...run, parameters: {...run.parameters, rounds: 3}}])).toContain('evaluation lengths');
    expect(incompatibility([run, {...run, experiment_id: 'r2'}])).toBe('');
    expect(incompatibility([run, {...run, experiment_id: 'n1'}])).toContain('protocols');
  });

  it('keeps descriptive comparisons available and labels matching parameters', async () => {
    const first: Run = {...run, status: 'completed', parameters: {...run.parameters, seed: 1},
      events: [{type: 'round', data: round}], summary: [{condition: 'Hedge', rounds: 1}]};
    const different: Run = {...first, id: 'different', parameters: {rounds: 5, seed: 2, strategy: 'uniform', fixed_action: 'Paper'}};
    const matched: Run = {...first, id: 'matched'};
    const candidates = [first, different, matched];
    vi.stubGlobal('fetch', vi.fn(async (path: string) => json(path.endsWith('/simulations') ? candidates :
      candidates.find(item => path.endsWith(`/${item.id}/results`)))));
    render(<MemoryRouter><ComparePage experiments={[experiment]}/></MemoryRouter>);
    const [one, two, three] = await screen.findAllByRole('checkbox');
    fireEvent.click(one);
    fireEvent.click(two);
    expect(await screen.findByText(/Descriptive comparison.*opponent strategy, seed/)).toBeTruthy();
    expect(screen.getByRole('heading', {name: 'Configurations'})).toBeTruthy();
    expect(screen.getByRole('button', {name: 'CSV'})).toBeTruthy();
    fireEvent.click(two);
    fireEvent.click(three);
    expect(await screen.findByText(/Matched run parameters/)).toBeTruthy();
  });

  it('loads recorded round data and steps through it without posting a run', async () => {
    const dataset: Dataset = {id: 'ollama-controlled', category: 'llm', title: 'Controlled Ollama Experiments',
      description: 'Recorded rounds', files: ['ollama_rps_results/rounds.csv']};
    const data: Historical = {id: dataset.id, sources: dataset.files, aggregate_only: false,
      rows: [{model: 'test', prefix: 0, strategy: 'uniform', b_reward: -1, paper_post: 0, entropy: 0, invalid: 0}],
      runs: [{condition: 'test', model: 'test', prefix: 0, strategy: 'uniform', rounds: [{...round, detail: {raw: 'Scissors'}}]}]};
    const fetcher = vi.fn(async (path: string, _init?: RequestInit) => {
      if (path.endsWith('/ollama-controlled')) return json(data);
      if (path.endsWith('/historical-results')) return json([dataset]);
      return json([]);
    });
    vi.stubGlobal('fetch', fetcher);
    const {Routes, Route} = await import('react-router-dom');
    render(<MemoryRouter initialEntries={['/replay/ollama-controlled']}><Routes>
      <Route path="/replay/:dataset" element={<ArchivePage/>}/></Routes></MemoryRouter>);
    await screen.findByText('Previously recorded research experiments');
    fireEvent.click(screen.getByRole('button', {name: 'Step to next round'}));
    expect(screen.getByRole('heading', {name: 'Round 1 inspector'})).toBeTruthy();
    expect(screen.getByRole('heading', {name: 'Raw Ollama response'})).toBeTruthy();
    expect(fetcher.mock.calls.every(call => call[1]?.method !== 'POST')).toBeTruthy();
  });
});
