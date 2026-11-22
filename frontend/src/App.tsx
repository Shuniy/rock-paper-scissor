import { lazy, Suspense, useEffect, useState } from 'react';
import { Archive, Atom, BookOpen, ChevronRight, FlaskConical, LayoutGrid, Menu, Moon,
  Play, ScanLine, Sun, Swords, X, Zap, BrainCircuit, Network } from 'lucide-react';
import { Link, NavLink, Route, Routes, useLocation, useParams } from 'react-router-dom';
import { api } from './api';
import type { Category, Experiment } from './api';
import { categories, findings, specialLinks } from './content';
import { Notice, PageHeading } from './UI';

const ExperimentPage = lazy(() => import('./ExperimentPage').then(module => ({default: module.ExperimentPage})));
const ArchivePage = lazy(() => import('./ResultsPages').then(module => ({default: module.ArchivePage})));
const ComparePage = lazy(() => import('./ResultsPages').then(module => ({default: module.ComparePage})));

const loop = [
  { label: 'Agent A', role: 'Learner', icon: BrainCircuit, text: 'A chooses an action using its current policy. In experiments with online learning, later feedback changes that policy.' },
  { label: 'Game', role: 'Actions → payoffs', icon: Swords, text: 'The game reveals both simultaneous moves and awards payoffs. In rock–paper–scissors, A’s reward and B’s reward add to zero.' },
  { label: 'Agent B', role: 'Strategist', icon: ScanLine, text: 'B chooses from the information allowed by its strategy. A planner may model future updates. A black-box opponent observes completed actions.' },
  { label: 'Learning update', role: 'Feedback → new policy', icon: Zap, text: 'An online learner uses the completed round to update its weights. A frozen LLM instead receives a changed prompt on its next turn.' },
];

function LearningLoop() {
  const [selected, select] = useState(0);
  return <div className="loop">
    <div className="loop-top"><span><Network size={15}/> THE INTERACTION LOOP</span><span className="loop-caption">Repeated game</span></div>
    <div className="loop-board">
      <svg className="loop-wires" viewBox="0 0 440 250" preserveAspectRatio="none" role="img" aria-label="Both agents choose actions, the game returns payoffs, and feedback updates Agent A.">
        <defs><marker id="mint-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M1,1 L8,5 L1,9" fill="none" stroke="var(--mint)" strokeWidth="1.4"/></marker>
          <marker id="amber-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M1,1 L8,5 L1,9" fill="none" stroke="var(--amber)" strokeWidth="1.4"/></marker></defs>
        <path d="M103 80 H149" className="wire mint-wire" markerEnd="url(#mint-arrow)"/>
        <path d="M337 80 H290" className="wire amber-wire" markerEnd="url(#amber-arrow)"/>
        <path d="M220 113 V161" className="wire mint-wire" markerEnd="url(#mint-arrow)"/>
        <path d="M147 195 H52 V114" className="wire mint-wire" strokeDasharray="4 4" markerEnd="url(#mint-arrow)"/>
        <path d="M272 48 V24 H388 V46" className="wire amber-wire" strokeDasharray="4 4" markerEnd="url(#amber-arrow)"/>
      </svg>
      {loop.map((node, index) => <button type="button" key={node.label} className={`loop-node node-${index} ${selected === index ? 'selected' : ''}`}
        onClick={() => select(index)} aria-pressed={selected === index}><node.icon size={18} strokeWidth={1.7}/><strong>{node.label}</strong><small>{node.role}</small></button>)}
    </div>
    <div className="loop-footer"><span className="legend-dot"/><p><strong>{loop[selected].label}.</strong> {loop[selected].text}</p></div>
  </div>;
}

const concepts = [
  ['External regret', 'Compare your play with the best fixed move in hindsight.', '/experiment/r1'],
  ['Swap regret', 'Allow a different replacement for each original move.', '/experiment/r2'],
  ['Neural learning', 'Predict the opponent’s next move from recent history.', '/experiment/n1'],
  ['Transformer policies', 'Optimize expected reward directly with a small Transformer.', '/experiment/n3'],
  ['Learning-aware planning', 'Anticipate changes to a learner’s future policy.', '/experiment/s1'],
  ['Black-box opponents', 'Estimate a hidden policy from public actions.', '/experiment/s5'],
  ['In-context LLM behaviour', 'Change the prompt history while keeping weights frozen.', '/experiment/l2'],
];

function Overview() {
  return <>
    <div className="hero">
      <div className="hero-copy">
        <div className="hero-tag"><FlaskConical size={15}/> INTERACTIVE RESEARCH LAB</div>
        <h1>Learning-Aware AI<br/><span>Simulation Lab</span></h1>
        <p className="hero-subtitle">An interactive exploration of regret learning, neural agents, and strategic LLM behaviour.</p>
        <p className="hero-description">Two agents play a repeated game. One learns from experience. The other can chase immediate wins or plan around how its moves will shape the learner’s future decisions.</p>
        <div className="hero-actions">
          <Link className="button primary" to="/experiment/r1"><Play size={17} fill="currentColor"/> Run Simulation</Link>
          <Link className="button secondary" to="/replay"><Archive size={17}/> Replay Research Results</Link>
        </div>
      </div>
      <LearningLoop/>
    </div>

    <section className="overview-section" aria-labelledby="modes-heading">
      <div className="section-heading"><div><p className="eyebrow">LAB MODES</p><h2 id="modes-heading">Choose how to explore</h2></div></div>
      <div className="mode-grid">
        <Link className="mode-card" to="/experiment/r1"><span className="mode-icon mint"><Play size={22}/></span><span>
          <strong>Run Simulation</strong><small>Configure an experiment and watch real decisions, updates, and rewards as it runs.</small></span><ChevronRight className="card-arrow" size={19}/></Link>
        <Link className="mode-card" to="/replay"><span className="mode-icon lavender"><Archive size={22}/></span><span>
          <strong>Replay Research Results</strong><small>Explore the recorded Ollama episodes and paired neural study immediately.</small></span><ChevronRight className="card-arrow" size={19}/></Link>
      </div>
    </section>

    <section className="overview-section">
      <div className="section-heading"><div><p className="eyebrow">EXPERIMENT CATALOG</p><h2>Explore the experiments</h2></div></div>
      <div className="category-grid">
        {Object.entries(categories).map(([id, category]) => <Link key={id} to={`/category/${id}`} className="category-card">
          <span className={`mode-icon ${category.color}`}><category.icon size={20}/></span>
          <span className="count">{category.number} experiments</span><strong>{category.title}</strong><small>{category.description}</small>
        </Link>)}
      </div>
    </section>

    <section className="overview-section two-columns">
      <div><p className="eyebrow">RESEARCH QUESTIONS</p><h2>What happens when opponents plan around learning?</h2><ol className="questions">
        <li>How does an agent learn from repeated interactions?</li>
        <li>Can an opponent influence what a learner prefers?</li>
        <li>Can regret-based objectives make learners more resistant?</li>
      </ol><Link className="text-link" to="/findings">Read the research findings <BookOpen size={15}/></Link></div>
      <div><p className="eyebrow">CONCEPT GUIDE</p><div className="concepts">
        {concepts.map(([title, description, href]) => <Link key={title} to={href}><span><strong>{title}</strong><small>{description}</small></span><ChevronRight size={17}/></Link>)}
      </div></div>
    </section>
  </>;
}

function CategoryPage({ experiments }: { experiments: Experiment[] }) {
  const { category } = useParams();
  const entry = category && Object.hasOwn(categories, category) ? categories[category as Category] : undefined;
  if (!entry) return <NotFound/>;
  return <>
    <PageHeading eyebrow="EXPERIMENT CATALOG" title={entry.title} description={entry.description}/>
    <div className="catalog-grid">{experiments.filter(exp => exp.category === category).map(exp =>
      <Link className="catalog-card" key={exp.id} to={`/experiment/${exp.id}`}>
        <span className={`mode-icon ${entry.color}`}><entry.icon size={19}/></span>
        <h2>{exp.title}</h2><p>{exp.description}</p><span className="text-link">Open experiment <ChevronRight size={17}/></span>
      </Link>)}</div>
    {(category === 'llm' || category === 'neural') && <div className="archive-callout"><Archive size={23}/><div><strong>Previously recorded research experiments</strong><p>Open the original study data without rerunning the experiments.</p></div>
      <Link className="button secondary" to={`/replay/${category === 'llm' ? 'ollama-controlled' : 'neural-validation'}`}>Browse results</Link></div>}
  </>;
}

function Findings() {
  return <>
    <PageHeading eyebrow="RESEARCH FINDINGS & LIMITATIONS" title="What the experiments found"
      description="The project tested when strategic opponents can profit from influencing a learner, and whether a swap penalty changes that vulnerability."/>
    <div className="finding-grid">{findings.map(([title, body], i) => <article className="panel finding" key={title}>
      <div className={`mode-icon ${i === 0 ? 'mint' : 'lavender'}`}>{i === 0 ? <Zap size={20}/> : <FlaskConical size={20}/>}</div><h2>{title}</h2><p>{body}</p></article>)}</div>
    <section className="panel limits"><h2>How to interpret these results</h2><ul>
      <li>Episode lengths, prompt settings, initializations, and opponent policies limit generalization.</li>
      <li>Paired neural runs match initial conditions and random draws. Adaptive opponents can generate different trajectories.</li>
      <li>Realized regret holds the observed opponent actions fixed, so it does not measure the effect of changing A’s policy against a reacting B.</li>
      <li>The expected-occupancy penalty is a training surrogate. Finite experiments establish no formal no-swap-regret guarantee.</li>
    </ul><div className="hero-actions"><Link className="button primary" to="/replay"><Archive size={17}/> Open recorded results</Link>
      <a className="button secondary" href="https://github.com/Shuniy/rock-paper-scissor#readme">Research README <BookOpen size={17}/></a></div></section>
  </>;
}

function ExperimentRoute({ experiments }: { experiments: Experiment[] }) {
  const { id } = useParams();
  const exp = experiments.find(item => item.id === id);
  return exp ? <ExperimentPage key={exp.id} exp={exp}/> : <NotFound/>;
}

function NotFound() {
  return <PageHeading eyebrow="SIMULATION LAB" title="Page unavailable" description="Open an experiment or the research archive from the lab navigation."><Link className="button primary" to="/">Overview</Link></PageHeading>;
}

export default function App() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [load, setLoad] = useState<'loading' | 'ready' | 'error'>('loading');
  const [connection, setConnection] = useState<'loading' | 'ready' | 'error'>('loading');
  const [version, setVersion] = useState(0);
  const [collapsed, setCollapsed] = useState(true);
  const [theme, setTheme] = useState(() => localStorage.getItem('lab-theme') || 'dark');
  const location = useLocation();
  useEffect(() => {
    let live = true;
    api<Experiment[]>('/experiments').then(value => { if (live) {setExperiments(value); setLoad('ready');} })
      .catch(() => { if (live) setLoad('error'); });
    return () => { live = false; };
  }, [version]);
  useEffect(() => {
    let live = true;
    let timer: number;
    let timeout: number;
    let controller: AbortController;
    async function checkHealth() {
      controller = new AbortController();
      timeout = window.setTimeout(() => controller.abort(), 5000);
      try {
        const health = await api<{status: string}>('/health', {cache: 'no-store', signal: controller.signal});
        if (live) setConnection(health.status === 'ok' ? 'ready' : 'error');
      } catch {
        if (live) setConnection('error');
      } finally {
        window.clearTimeout(timeout);
        if (live) timer = window.setTimeout(checkHealth, 15000);
      }
    }
    void checkHealth();
    return () => {
      live = false;
      window.clearTimeout(timer);
      window.clearTimeout(timeout);
      controller.abort();
    };
  }, []);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem('lab-theme', theme);
  }, [theme]);
  useEffect(() => { window.scrollTo(0, 0); setCollapsed(true); }, [location.pathname]);
  const parts = location.pathname.split('/').filter(Boolean);
  const currentExperiment = experiments.find(exp => exp.id === parts[1]);
  const currentCategory = parts[0] === 'category' ? parts[1] as Category : currentExperiment?.category;
  const breadcrumb = currentCategory ? categories[currentCategory]?.title : {
    replay: 'Replay Research Results', compare: 'Experiment Comparison', findings: 'Research Findings'
  }[parts[0]] || 'Overview';
  return <div className="shell">
    <a className="skip-link" href="#lab-content">Skip to content</a>
    {!collapsed && <button type="button" className="screen-overlay" aria-label="Close navigation" onClick={() => setCollapsed(true)}/>}
    <aside id="lab-sidebar" className={`sidebar ${collapsed ? '' : 'expanded'}`}>
      <Link className="brand" to="/"><div className="brand-mark"><Atom size={24} strokeWidth={1.8}/></div>
        <span><strong>Simulation Lab</strong><small>LEARNING-AWARE AI</small></span></Link>
      <div className="side-nav">
        <p className="nav-label">WORKSPACE</p>
        <NavLink to="/" end><LayoutGrid size={18}/><span>Overview</span></NavLink>
        <NavLink to="/replay"><Archive size={18}/><span>Replay Research Results</span></NavLink>
        <p className="nav-label">EXPERIMENTS</p>
        {Object.entries(categories).map(([id, category]) => <NavLink className={() => currentCategory === id ? 'active' : ''}
          key={id} to={`/category/${id}`}><category.icon size={18}/><span>{category.title}</span></NavLink>)}
        <p className="nav-label">ANALYSIS</p>
        {specialLinks.map(link => <NavLink key={link.href} to={link.href}><link.icon size={18}/><span>{link.title}</span></NavLink>)}
      </div>
      <div className="sidebar-footer"><span className={`status-dot ${connection}`}/><div><strong>{connection === 'ready' ? 'Local lab online' : connection === 'error' ? 'API unavailable' : 'Connecting to lab'}</strong><small>Runs on your machine</small></div></div>
    </aside>
    <div className="workspace">
      <header className="topbar">
        <div className="breadcrumb"><button className="icon-button menu-toggle" aria-expanded={!collapsed} aria-controls="lab-sidebar" aria-label={collapsed ? 'Open navigation' : 'Close navigation'} onClick={() => setCollapsed(!collapsed)}>{collapsed ? <Menu size={20}/> : <X size={20}/>}</button>
          <span>Workspace</span><ChevronRight size={14}/><strong>{breadcrumb}</strong></div>
        <button type="button" className="icon-button theme-toggle" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}>{theme === 'dark' ? <Sun size={18}/> : <Moon size={18}/>}</button>
      </header>
      <main id="lab-content" className="main-content">
        {load !== 'ready' && <Notice danger={load === 'error'}>{load === 'error' ?
          <>The local backend could not be reached. Start the Python API and <button className="text-button" onClick={() => {setLoad('loading');setVersion(i => i + 1);}}>retry</button>.</> : 'Connecting to the local API…'}</Notice>}
        <Suspense fallback={<div className="panel empty-state" role="status">Loading the lab view…</div>}><Routes>
          <Route path="/" element={<Overview/>}/>
          <Route path="/category/:category" element={<CategoryPage experiments={experiments}/>}/>
          <Route path="/experiment/:id" element={load === 'ready' ? <ExperimentRoute experiments={experiments}/> : null}/>
          <Route path="/replay" element={<ArchivePage/>}/>
          <Route path="/replay/:dataset" element={<ArchivePage/>}/>
          <Route path="/compare" element={<ComparePage experiments={experiments}/>}/>
          <Route path="/findings" element={<Findings/>}/>
          <Route path="*" element={<NotFound/>}/>
        </Routes></Suspense>
        <footer className="footer"><Atom size={15}/> Learning-Aware AI Simulation Lab <span>Preliminary research · local workspace</span></footer>
      </main>
    </div>
  </div>;
}
