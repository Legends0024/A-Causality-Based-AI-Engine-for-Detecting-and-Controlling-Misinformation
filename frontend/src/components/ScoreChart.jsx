import React from 'react';

const ScoreChart = ({ scoreHistory }) => {
  if (!scoreHistory || scoreHistory.length < 2) return null;

  const W = 200, H = 80;
  const PAD = { top: 10, right: 10, bottom: 24, left: 36 };
  const iW = W - PAD.left - PAD.right;
  const iH = H - PAD.top - PAD.bottom;

  const maxS = Math.max(...scoreHistory);
  const minS = Math.min(...scoreHistory);
  const range = maxS - minS || 1;

  const px = i => PAD.left + (i / (scoreHistory.length - 1)) * iW;
  const py = s => PAD.top + iH - ((s - minS) / range) * iH;

  const points = scoreHistory.map((s, i) => `${px(i)},${py(s)}`).join(' ');
  const areaClose = `${px(scoreHistory.length - 1)},${PAD.top + iH} ${px(0)},${PAD.top + iH}`;

  const reductionPct = ((scoreHistory[0] - scoreHistory[scoreHistory.length - 1]) / scoreHistory[0] * 100).toFixed(1);

  return (
    <div className="glass-card" style={{ padding: '20px 20px 12px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <h3 style={{ fontSize: '0.95rem', fontWeight: 600 }}>Containment Trajectory</h3>
        <span style={{ fontSize: '0.8rem', color: 'var(--accent-emerald)', fontWeight: 600 }}>
          ↓ {reductionPct}% total
        </span>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: '140px', overflow: 'visible' }}>
        <defs>
          <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#10b981" stopOpacity="0.35" />
            <stop offset="100%" stopColor="#10b981" stopOpacity="0" />
          </linearGradient>
        </defs>

        {/* Y-axis gridlines */}
        {[0, 0.5, 1].map(t => {
          const y = PAD.top + iH * (1 - t);
          const val = (minS + t * range).toFixed(2);
          return (
            <g key={t}>
              <line x1={PAD.left} y1={y} x2={PAD.left + iW} y2={y}
                stroke="rgba(255,255,255,0.06)" strokeWidth="1" />
              <text x={PAD.left - 4} y={y} fill="rgba(255,255,255,0.3)"
                fontSize="5" textAnchor="end" dominantBaseline="middle">
                {val}
              </text>
            </g>
          );
        })}

        {/* X-axis labels */}
        {scoreHistory.map((_, i) => (
          <text key={i} x={px(i)} y={PAD.top + iH + 10}
            fill="rgba(255,255,255,0.3)" fontSize="5" textAnchor="middle">
            {i === 0 ? 'Base' : `K${i}`}
          </text>
        ))}

        {/* Area fill */}
        <polygon points={`${points} ${areaClose}`} fill="url(#areaGrad)" />

        {/* Line */}
        <polyline fill="none" stroke="#10b981" strokeWidth="1.2" points={points}
          style={{ filter: 'drop-shadow(0 0 3px rgba(16,185,129,0.6))' }} />

        {/* Data points */}
        {scoreHistory.map((s, i) => (
          <g key={i}>
            <circle cx={px(i)} cy={py(s)} r="2.5" fill="white" stroke="#10b981" strokeWidth="1" />
            <title>Step {i}: {s.toFixed(4)}</title>
          </g>
        ))}
      </svg>
    </div>
  );
};

export default ScoreChart;