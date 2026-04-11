import React from 'react';

const ScoreChart = ({ scoreHistory }) => {
  const width = 100; // Using viewBox for responsiveness
  const height = 40;
  const maxScore = Math.max(...scoreHistory);
  const minScore = Math.min(...scoreHistory);
  const range = maxScore - minScore || 1;
  const padding = 5;
  const innerWidth = width - padding * 2;
  const innerHeight = height - padding * 2;
  
  const points = scoreHistory.map((score, i) => {
    const x = padding + (i / (scoreHistory.length - 1)) * innerWidth;
    const y = height - padding - ((score - minScore) / range) * innerHeight;
    return `${x},${y}`;
  }).join(' ');

  const areaPoints = `${padding},${height - padding} ${points} ${padding + innerWidth},${height - padding}`;

  return (
    <div className="glass-card p-6" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <h3 className="mb-4" style={{ fontSize: '1.1rem' }}>Containment Trajectory</h3>
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', position: 'relative' }}>
        <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" style={{ width: '100%', height: '120px', overflow: 'visible' }}>
          <defs>
            <linearGradient id="chartGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--accent-emerald)" stopOpacity="0.3" />
              <stop offset="100%" stopColor="var(--accent-emerald)" stopOpacity="0" />
            </linearGradient>
          </defs>
          <polygon points={areaPoints} fill="url(#chartGradient)" />
          <polyline fill="none" stroke="var(--accent-emerald)" strokeWidth="0.8" points={points} style={{ filter: 'drop-shadow(0 0 2px var(--accent-emerald-glow))' }} />
          
          {scoreHistory.map((score, i) => {
            const x = padding + (i / (scoreHistory.length - 1)) * innerWidth;
            const y = height - padding - ((score - minScore) / range) * innerHeight;
            return (
              <circle key={i} cx={x} cy={y} r="1" fill="#fff" stroke="var(--accent-emerald)" strokeWidth="0.5" />
            );
          })}
        </svg>
      </div>
    </div>
  );
};

export default ScoreChart;