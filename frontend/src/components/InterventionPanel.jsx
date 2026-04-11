import React from 'react';

const InterventionPanel = ({ interventionNodes, reductionPct, randomReduction, baselineScore, finalScore, prediction }) => {
  const greaterThanRandom = reductionPct > randomReduction;
  const isRealNews = prediction === 'non-rumour' || (!prediction && interventionNodes.length === 0);

  return (
    <div className="glass-card" style={{ padding: '20px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <h3 style={{ fontSize: '0.95rem', fontWeight: 600 }}>Optimal Debunking Targets</h3>
        <span className="chip chip-emerald" style={{ fontSize: '0.7rem', padding: '2px 8px' }}>
          via GAT
        </span>
      </div>

      {/* Target nodes */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '20px' }}>
        {interventionNodes.length === 0 ? (
          <div style={{
            padding: '12px 14px',
            background: isRealNews ? 'rgba(16,185,129,0.08)' : 'rgba(255,255,255,0.04)',
            borderRadius: '8px',
            border: `1px solid ${isRealNews ? 'rgba(16,185,129,0.25)' : 'var(--border-color)'}`,
          }}>
            <p style={{ fontSize: '0.8rem', color: isRealNews ? 'var(--accent-emerald)' : 'var(--text-muted)', lineHeight: 1.5 }}>
              {isRealNews
                ? '✓ No intervention needed — content appears credible. Monitor for unusual spread velocity.'
                : 'No high-risk nodes identified in this cascade.'}
            </p>
          </div>
        ) : interventionNodes.map((node, i) => (
          <div key={i} style={{
            display: 'flex', alignItems: 'center', gap: '12px',
            padding: '10px 14px', background: 'var(--bg-dark)',
            borderRadius: '8px', border: '1px solid var(--border-color)',
          }}>
            <div style={{
              width: '24px', height: '24px', borderRadius: '50%',
              background: 'var(--accent-emerald)', color: '#000',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontWeight: 700, fontSize: '0.7rem', flexShrink: 0,
              boxShadow: '0 0 8px rgba(16,185,129,0.5)',
            }}>
              {i + 1}
            </div>
            <div className="mono" style={{ flex: 1, fontSize: '0.875rem' }}>Node {node}</div>
            <div style={{ fontSize: '0.7rem', color: 'var(--accent-amber)', fontWeight: 600 }}>
              Priority {i === 0 ? 'CRITICAL' : i === 1 ? 'HIGH' : 'MED'}
            </div>
          </div>
        ))}
      </div>

      {/* Comparison bars */}
      <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '16px' }}>
        <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Strategy Comparison
        </p>

        {[
          { label: 'Causal (AI)', pct: reductionPct, color: 'var(--accent-emerald)' },
          { label: 'Random', pct: randomReduction, color: 'rgba(255,255,255,0.2)' },
        ].map(({ label, pct, color }) => (
          <div key={label} style={{ marginBottom: '10px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{label}</span>
              <span style={{ fontSize: '0.8rem', fontWeight: 600, color }}>{pct.toFixed(1)}%</span>
            </div>
            <div style={{ background: 'rgba(255,255,255,0.05)', borderRadius: '4px', height: '6px', overflow: 'hidden' }}>
              <div style={{
                height: '100%', borderRadius: '4px',
                background: color, width: `${Math.min(pct, 100)}%`,
                transition: 'width 0.8s ease',
                boxShadow: label === 'Causal (AI)' ? '0 0 8px rgba(16,185,129,0.5)' : 'none',
              }} />
            </div>
          </div>
        ))}

        <div style={{
          marginTop: '12px', padding: '8px 12px',
          background: greaterThanRandom ? 'rgba(16,185,129,0.08)' : 'rgba(239,68,68,0.08)',
          border: `1px solid ${greaterThanRandom ? 'rgba(16,185,129,0.2)' : 'rgba(239,68,68,0.2)'}`,
          borderRadius: '8px', fontSize: '0.78rem',
          color: greaterThanRandom ? 'var(--accent-emerald)' : 'var(--accent-rose)',
        }}>
          {greaterThanRandom
            ? `✓ AI outperforms random by +${(reductionPct - randomReduction).toFixed(1)}%`
            : `⚠ Similar to random — model may need training data`}
        </div>
      </div>
    </div>
  );
};

export default InterventionPanel;