import React from 'react';

const CascadeList = ({ cascades, selectedId, onSelect }) => {
  return (
    <div>
      <h3 className="text-secondary mb-4" style={{ fontSize: '0.875rem', textTransform: 'uppercase', letterSpacing: '1px' }}>Available Cascades</h3>
      {cascades.length === 0 ? (
        <p className="text-muted" style={{ fontSize: '0.875rem' }}>No cascades loaded.</p>
      ) : (
        <div className="flex flex-col gap-2">
          {cascades.map(cascade => (
            <div 
              key={cascade.id} 
              onClick={() => onSelect(cascade.id)}
              className={`glass-card p-4 flex justify-between items-center ${selectedId === cascade.id ? 'active' : ''}`}
              style={{ 
                cursor: 'pointer', 
                border: selectedId === cascade.id ? '1px solid var(--accent-indigo)' : '1px solid var(--border-color)',
                boxShadow: selectedId === cascade.id ? '0 0 15px rgba(99, 102, 241, 0.1)' : 'none',
                background: selectedId === cascade.id ? 'rgba(99, 102, 241, 0.05)' : 'var(--bg-card)'
              }}
            >
              <div>
                <div style={{ fontWeight: 500, marginBottom: '4px' }}>Cascade #{cascade.id}</div>
                <div className="text-muted" style={{ fontSize: '0.75rem' }}>
                  {cascade.nodes} Nodes • {cascade.edges} Edges
                </div>
              </div>
              <span className={`chip ${cascade.label === 'rumour' ? 'chip-rose' : 'chip-indigo'}`} style={{ fontSize: '0.7rem', padding: '2px 8px' }}>
                {cascade.label === 'rumour' ? 'RMR' : 'NON'}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default CascadeList;