import React from 'react';

const InterventionPanel = ({ interventionNodes, reductionPct, randomReduction }) => {
  return (
    <div className="glass-card p-6">
      <div className="flex items-center gap-2 mb-4">
        <h3 style={{ fontSize: '1.1rem' }}>Optimal Debunking Targets</h3>
        <span className="chip chip-emerald" style={{ marginLeft: 'auto' }}>Calculated via GNN</span>
      </div>
      
      <p className="text-secondary mb-4" style={{ fontSize: '0.875rem' }}>
        The Causal Engine recommends isolating the following core nodes to maximize containment.
      </p>
      
      <div className="flex flex-col gap-2">
        {interventionNodes.map((node, i) => (
          <div key={i} className="flex items-center p-3" style={{ background: 'var(--bg-dark)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
            <div style={{ width: '28px', height: '28px', borderRadius: '50%', background: 'var(--accent-emerald)', color: 'black', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', fontSize: '0.75rem', marginRight: '12px', boxShadow: '0 0 10px var(--accent-emerald-glow)' }}>
              {i + 1}
            </div>
            <div className="mono">Node {node}</div>
            <div className="text-muted ml-auto" style={{ fontSize: '0.75rem' }}>Priority High</div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default InterventionPanel;