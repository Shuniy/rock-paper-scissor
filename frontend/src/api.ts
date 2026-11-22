export type Value = string | number;
export type Parameters = Record<string, Value>;
export type Category = 'regret' | 'neural' | 'strategy' | 'llm';
export type Field = {
  key: string; label: string; default: Value; kind: 'number' | 'select' | 'model' | 'optional_model' | 'text';
  min?: number; max?: number; step?: number; options?: Record<string, string>; hint?: string;
  when?: Record<string, Value[]>;
};
export type Experiment = {
  id: string; title: string; category: Category; description: string; game: string; fields: Field[];
};
export type Round = {
  run_id: string; experiment_id: string; condition: string; round: number; a: string; b: string;
  reward_a: number; reward_b: number; cumulative_a: number; cumulative_b: number;
  average_a: number; average_b: number; external: number; swap: number;
  frequency: number[]; policy?: number[]; phase?: string; history?: unknown[]; detail: Record<string, unknown>;
};
export type Stage = { condition: string; step: number; total: number; loss?: number; expected_a?: number; swap_surrogate?: number };
export type RunEvent =
  | { type: 'round'; data: Round }
  | { type: 'warmup' | 'training'; data: Stage }
  | { type: 'started' | 'completed' | 'cancelled' | 'failed'; data: { message?: string } };
export type RunStatus = 'queued' | 'running' | 'cancelling' | 'completed' | 'cancelled' | 'failed';
export type RunMeta = {
  id: string; experiment_id: string; game: string; parameters: Parameters; status: RunStatus;
  created_at: string; completed_rounds: number; total_rounds: number; model_calls: number | null;
  conditions: string[]; message?: string;
};
export type TabularRow = Record<string, string | number | null>;
export type Run = RunMeta & { events: RunEvent[]; summary: TabularRow[]; paired?: TabularRow[] };
export type Dataset = {
  id: string; title: string; description: string; category: Category; files: string[];
};
export type Historical = {
  id: string; sources: string[]; aggregate_only: boolean; rows: TabularRow[]; paired?: TabularRow[];
  runs?: {condition: string; model: string; prefix: number; strategy: string; rounds: Round[]}[];
};

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch('/api' + path, init);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const details = Array.isArray(body.detail) ? body.detail.map((issue: {msg: string}) => issue.msg).join(' ') : body.detail;
    throw new Error(details || `The lab request failed (${response.status}).`);
  }
  return response.json();
}

export function active(status?: RunStatus) {
  return status === 'queued' || status === 'running' || status === 'cancelling';
}

export function rowsOf(events: RunEvent[]) {
  return events.flatMap(event => event.type === 'round' ? [event.data] : []);
}

export function exportUrl(id: string, format: string) {
  return `/api/simulations/${id}/export?format=${format}`;
}

export function download(name: string, data: string, type: string) {
  const url = URL.createObjectURL(new Blob([data], {type}));
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function downloadCsv(name: string, rows: TabularRow[]) {
  const keys = [...new Set(rows.flatMap(Object.keys))];
  const cell = (value: unknown) => {
    let str = String(value ?? '');
    if (typeof value === 'string' && /^[=+@-]/.test(str.trimStart())) str = "'" + str;
    return '"' + str.replaceAll('"', '""') + '"';
  };
  download(name, [keys.map(cell), ...rows.map(row => keys.map(key => cell(row[key])))].map(row => row.join(',')).join('\n'), 'text/csv');
}
