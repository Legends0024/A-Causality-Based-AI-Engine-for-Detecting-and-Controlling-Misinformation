import React from 'react';

const StatsCards = ({ label, nodes, reductionPct, improvement }) => {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px', marginBottom: '20px' }}>
      <div style={{ background: '#0a0f1e', padding: '10px', border: '1px solid #1e293b', borderRadius: '5px' }}>
        <h4>Rumour Verdict</h4>
        <p>{label}</p>
      </div>
      <div style={{ background: '#0a0f1e', padding: '10px', border: '1px solid #1e293b', borderRadius: '5px' }}>
        <h4>Total Nodes</h4>
        <p>{nodes}</p>
      </div>
      <div style={{ background: '#0a0f1e', padding: '10px', border: '1px solid #1e293b', borderRadius: '5px' }}>
        <h4>Spread Reduced</h4>
        <p style={{ color: '#10b981' }}>{reductionPct.toFixed(1)}%</p>
      </div>
      <div style={{ background: '#0a0f1e', padding: '10px', border: '1px solid #1e293b', borderRadius: '5px' }}>
        <h4>Vs Random</h4>
        <p style={{ color: '#f59e0b' }}>{improvement.toFixed(1)}%</p>
      </div>
    </div>
  );
};

export default StatsCards;