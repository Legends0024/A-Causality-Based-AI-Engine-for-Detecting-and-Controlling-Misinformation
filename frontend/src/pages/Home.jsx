import React from 'react';
import { useNavigate } from 'react-router-dom';

const PIPELINE = [
  { icon: '📡', title: 'Raw Input', sub: 'News text / URL / screenshot', color: '#334155', border: '#475569' },
  { icon: '🤖', title: 'BERT Classifier', sub: 'Pretrained fake-news model', color: '#1e3a5f', border: '#2563eb' },
  { icon: '📋', title: 'Rule Engine', sub: 'Phrase & plausibility checks', color: '#14532d', border: '#16a34a' },
  { icon: '🌐', title: 'Live Web Check', sub: 'Google News RSS verification', color: '#3b1f6b', border: '#7c3aed' },
  { icon: '🕸️', title: 'Causal Graph', sub: 'GAT cascade model', color: '#7c2d12', border: '#ea580c' },
  { icon: '🎯', title: 'Intervention', sub: 'Greedy optimizer (K nodes)', color: '#164e63', border: '#0891b2' },
];

const ArchitecturePipeline = () => (
  <div style={{ marginTop: '64px', padding: '40px', background: 'rgba(16,23,42,0.4)', borderRadius: '20px', border: '1px solid var(--border-color)' }}>
    <h3 style={{ textAlign: 'center', marginBottom: '40px', color: 'var(--text-secondary)', fontSize: '0.875rem', textTransform: 'uppercase', letterSpacing: '2px' }}>
      System Architecture Pipeline
    </h3>
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flexWrap: 'wrap', gap: '0' }}>
      {PIPELINE.map((step, i) => (
        <React.Fragment key={i}>
          <div style={{
            background: step.color, border: `1px solid ${step.border}`,
            borderRadius: '12px', padding: '20px 16px', width: '150px',
            textAlign: 'center', boxShadow: `0 0 20px ${step.border}22`,
          }}>
            <div style={{ fontSize: '1.8rem', marginBottom: '8px' }}>{step.icon}</div>
            <div style={{ fontWeight: 600, fontSize: '0.875rem', color: '#fff', marginBottom: '4px' }}>{step.title}</div>
            <div style={{ fontSize: '0.7rem', color: 'rgba(255,255,255,0.6)', lineHeight: 1.4 }}>{step.sub}</div>
          </div>
          {i < PIPELINE.length - 1 && (
            <div style={{ color: 'rgba(255,255,255,0.3)', fontSize: '1.5rem', margin: '0 4px', userSelect: 'none' }}>→</div>
          )}
        </React.Fragment>
      ))}
    </div>
  </div>
);

const FEATURES = [
  { icon: '🧠', title: 'BERT + TF-IDF', desc: 'Pretrained transformer classifier with TF-IDF fallback. Far more accurate than keyword matching alone.' },
  { icon: '🌐', title: 'Live Fact-Check', desc: 'Queries Google News RSS in real time when the model is uncertain, leveraging current news coverage.' },
  { icon: '📊', title: 'Plausibility Guard', desc: 'Catches numerically impossible claims like "Sensex +9000 pts" or "RBI cuts rate 40%".' },
  { icon: '🕸️', title: 'Graph Attention Network', desc: 'GAT model assigns spread-risk scores to every node in the social network cascade.' },
  { icon: '🎯', title: 'Greedy Optimizer', desc: 'Finds the minimum set of nodes to suppress for maximum misinformation containment.' },
  { icon: '📉', title: 'Beats Random by Design', desc: 'Causal intervention provably outperforms random node removal — shown in every analysis run.' },
];

const Home = () => {
  const navigate = useNavigate();
  return (
    <div style={{ minHeight: '100vh', position: 'relative', overflowX: 'hidden' }}>

      {/* Background orb */}
      <div style={{
        position: 'fixed', top: '40%', left: '50%', transform: 'translate(-50%, -50%)',
        width: '70vw', height: '70vw', pointerEvents: 'none', zIndex: 0,
        background: 'radial-gradient(circle, rgba(16,185,129,0.04) 0%, transparent 70%)',
      }} />

      <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '80px 32px 120px', position: 'relative', zIndex: 1 }}>

        {/* Hero */}
        <div style={{ textAlign: 'center', marginBottom: '24px' }}>
          <div style={{ display: 'flex', justifyContent: 'center', gap: '12px', marginBottom: '32px', flexWrap: 'wrap' }}>
            <span className="chip chip-indigo">BERT + GAT Fusion</span>
            <span className="chip chip-emerald">Causal AI</span>
            <span className="chip chip-amber">Live Fact-Check</span>
          </div>

          <h1 style={{ fontSize: 'clamp(2rem, 5vw, 3.75rem)', lineHeight: 1.15, fontWeight: 800, marginBottom: '24px', letterSpacing: '-0.03em' }}>
            Causal intervention engine for{' '}
            <span style={{ color: 'var(--accent-emerald)' }}>misinformation containment</span>
          </h1>

          <p style={{ fontSize: '1.2rem', color: 'var(--text-secondary)', lineHeight: 1.7, maxWidth: '780px', margin: '0 auto 40px' }}>
            Don't just predict where fake news spreads — ask <em>"if we debunk node X at time T, what's the counterfactual spread?"</em> Build an AI that runs causal interventions on live social graphs to find the minimum set of nodes that collapses a rumour's reach.
          </p>

          <div style={{ display: 'flex', gap: '16px', justifyContent: 'center', flexWrap: 'wrap' }}>
            <button
              className="btn-primary"
              style={{ fontSize: '1.05rem', padding: '14px 36px' }}
              onClick={() => navigate('/dashboard')}
            >
              Launch Engine Dashboard →
            </button>
            <a
              href="https://github.com/Legends0024/A-Causality-Based-AI-Engine-for-Detecting-and-Controlling-Misinformation"
              target="_blank" rel="noopener noreferrer"
              className="btn-outline"
              style={{ padding: '14px 28px', display: 'inline-flex', alignItems: 'center', gap: '8px' }}
            >
              ⭐ GitHub
            </a>
          </div>
        </div>

        {/* Architecture */}
        <ArchitecturePipeline />

        {/* Feature grid */}
        <div style={{ marginTop: '80px' }}>
          <h2 style={{ textAlign: 'center', marginBottom: '40px', fontSize: '1.75rem', fontWeight: 700 }}>
            What makes this different
          </h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
            {FEATURES.map((f, i) => (
              <div key={i} className="glass-card" style={{ padding: '24px' }}>
                <div style={{ fontSize: '2rem', marginBottom: '12px' }}>{f.icon}</div>
                <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '8px' }}>{f.title}</h3>
                <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>{f.desc}</p>
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
};

export default Home;