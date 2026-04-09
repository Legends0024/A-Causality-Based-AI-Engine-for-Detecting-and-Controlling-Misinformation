import React from 'react';

const InterventionPanel = ({ interventionNodes, reductionPct, randomReduction }) => {
  const improvement = reductionPct - randomReduction;
  return (
    <div>
      <h3>Intervention Results</h3>
      <p>Debunking {interventionNodes.length} nodes reduces projected spread by {reductionPct.toFixed(1)}%</p>
      <p>Random removal: {randomReduction.toFixed(1)}% | Our method: {reductionPct.toFixed(1)}% | Gain: +{improvement.toFixed(1)}%</p>
      <div>
        {interventionNodes.map((node, i) => (
          <span key={i} style={{ background: '#10b981', color: 'white', padding: '5px', margin: '2px', borderRadius: '3px' }}>
            {i+1}. {node}
          </span>
        ))}
      </div>
    </div>
  );
};

export default InterventionPanel;