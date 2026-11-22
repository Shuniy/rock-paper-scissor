// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Dataset, Experiment, Historical, Round, Run, RunEvent, RunMeta } from './api';
import { comparisonData } from './Dashboard';
import { estimateModelCalls, ExperimentPage, visibleFields } from './ExperimentPage';
import { ArchivePage, incompatibility } from './ResultsPages';

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

beforeEach(() => {
  TestEventSource.instances = [];
  vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockReturnValue(new DOMRect(0, 0, 450, 235));
  vi.stubGlobal('EventSource', TestEventSource);
  vi.stubGlobal('ResizeObserver', class {observe(){} unobserve(){} disconnect(){}});
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

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
