import React from 'react';

const ScoreChart = ({ scoreHistory }) => {
  const width = 400;
  const height = 200;
  const padding = 40;
  const maxScore = Math.max(...scoreHistory);
  const minScore = Math.min(...scoreHistory);
  const range = maxScore - minScore || 1;
  const points = scoreHistory.map((score, i) => {
    const x = padding + (i / (scoreHistory.length - 1)) * (width - 2 * padding);
    const y = height - padding - ((score - minScore) / range) * (height - 2 * padding);
    return `${x},${y}`;
  }).join(' ');
  const reduction = scoreHistory.length > 1 ? ((scoreHistory[0] - scoreHistory[scoreHistory.length - 1]) / scoreHistory[0] * 100).toFixed(1) : 0;
  return (
    <div>
      <h3>Spread Score Reduction</h3>
      <svg width={width} height={height} style={{ border: '1px solid #1e293b', background: '#020817' }}>
        <polyline fill="none" stroke="#10b981" strokeWidth="2" points={points} />
        {scoreHistory.map((score, i) => {
          const x = padding + (i / (scoreHistory.length - 1)) * (width - 2 * padding);
          const y = height - padding - ((score - minScore) / range) * (height - 2 * padding);
          return <circle key={i} cx={x} cy={y} r="4" fill="#10b981" />;
        })}
        <text x={width - padding} y={padding} fill="#10b981" fontSize="12">{reduction}% reduction</text>
      </svg>
    </div>
  );
};

export default ScoreChart;