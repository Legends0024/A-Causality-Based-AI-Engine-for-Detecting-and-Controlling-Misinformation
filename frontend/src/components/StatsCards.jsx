import React from 'react';

const StatsCards = ({ label, nodes, reductionPct, improvement }) => {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
      <div className="glass-card p-6">
        <div className="text-muted mb-2" style={{ fontSize: '0.875rem' }}>Network Size</div>
        <div style={{ fontSize: '2rem', fontWeight: 600 }}>{nodes}</div>
        <div className="text-emerald mt-2" style={{ fontSize: '0.875rem' }}>Targetable Nodes</div>
      </div>
      
      <div className="glass-card p-6">
        <div className="text-muted mb-2" style={{ fontSize: '0.875rem' }}>Label Status</div>
        <div style={{ fontSize: '2rem', fontWeight: 600, color: label === 'rumour' ? 'var(--accent-rose)' : 'var(--accent-indigo)', textTransform: 'capitalize' }}>
          {label}
        </div>
        <div className="text-secondary mt-2" style={{ fontSize: '0.875rem' }}>Detected category</div>
      </div>
      
      <div className="glass-card p-6" style={{ position: 'relative', overflow: 'hidden' }}>
        <div style={{ position: 'absolute', top: '-10px', right: '-10px', width: '50px', height: '50px', background: 'var(--accent-emerald-glow)', filter: 'blur(20px)' }}></div>
        <div className="text-muted mb-2" style={{ fontSize: '0.875rem' }}>Threat Reduction</div>
        <div className="text-emerald" style={{ fontSize: '2rem', fontWeight: 600 }}>{reductionPct.toFixed(1)}%</div>
        <div className="text-secondary mt-2" style={{ fontSize: '0.875rem' }}>Spread probability drop</div>
      </div>
      
      <div className="glass-card p-6" style={{ position: 'relative', overflow: 'hidden' }}>
        <div style={{ position: 'absolute', top: '-10px', right: '-10px', width: '50px', height: '50px', background: 'rgba(99, 102, 241, 0.2)', filter: 'blur(20px)' }}></div>
        <div className="text-muted mb-2" style={{ fontSize: '0.875rem' }}>AI Advantage</div>
        <div className="text-indigo" style={{ fontSize: '2rem', fontWeight: 600 }}>+{improvement.toFixed(1)}%</div>
        <div className="text-secondary mt-2" style={{ fontSize: '0.875rem' }}>vs random targeting</div>
      </div>
    </div>
  );
};

export default StatsCards;