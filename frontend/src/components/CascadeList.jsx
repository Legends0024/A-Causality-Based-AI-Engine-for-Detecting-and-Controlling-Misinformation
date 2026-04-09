import React from 'react';

const CascadeList = ({ cascades, selectedId, onSelect }) => {
  const getLabelColor = (label) => {
    if (label === 'false') return '#ef4444';
    if (label === 'true') return '#10b981';
    if (label === 'unverified') return '#f59e0b';
    return '#3b82f6';
  };
  return (
    <div style={{ height: '400px', overflowY: 'scroll', border: '1px solid #1e293b', padding: '10px' }}>
      {cascades.map(cascade => (
        <div key={cascade.id} onClick={() => onSelect(cascade.id)} style={{
          padding: '10px',
          marginBottom: '5px',
          background: selectedId === cascade.id ? '#0a0f1e' : '#020817',
          borderLeft: selectedId === cascade.id ? '4px solid #10b981' : '4px solid transparent',
          cursor: 'pointer'
        }}>
          <div>Cascade {cascade.id}</div>
          <div style={{ color: getLabelColor(cascade.label) }}>{cascade.label}</div>
          <div>{cascade.nodes} nodes, {cascade.edges} edges</div>
        </div>
      ))}
    </div>
  );
};

export default CascadeList;