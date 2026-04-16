import React from 'react';

const Card = ({ label, value, sub, color = 'var(--text-primary)', glow }) => (
  <div className="glass-card" style={{ padding: '20px 24px', position: 'relative', overflow: 'hidden' }}>
    {glow && (
      <div style={{ position: 'absolute', top: -10, right: -10, width: 50, height: 50, background: glow, filter: 'blur(20px)', borderRadius: '50%' }} />
    )}
    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '8px', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</div>
    <div style={{ fontSize: '1.9rem', fontWeight: 700, color, lineHeight: 1 }}>{value}</div>
    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '6px' }}>{sub}</div>
  </div>
);

const StatsCards = ({ label, nodes, reductionPct, improvement, graphFakeProb }) => (
  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
    <Card
      label="Network Size"
      value={nodes}
      sub="Targetable nodes"
      color="var(--text-primary)"
    />
    <Card
      label="Veracity Verdict"
      value={label.split(' ').pop().toUpperCase()}
      sub={label}
      color={['Misinformation', 'Likely Misinformation'].includes(label) ? 'var(--accent-rose)' : 'var(--accent-indigo)'}
    />
    <Card
      label="Spread Reduced"
      value={`${reductionPct.toFixed(1)}%`}
      sub="vs baseline cascade"
      color="var(--accent-emerald)"
      glow="rgba(16,185,129,0.3)"
    />
    <Card
      label="AI Advantage"
      value={`+${improvement.toFixed(1)}%`}
      sub="vs random targeting"
      color="var(--accent-indigo)"
      glow="rgba(99,102,241,0.25)"
    />
  </div>
);

export default StatsCards;