import React, { useState, useMemo } from 'react';

const CascadeList = ({ cascades, selectedId, onSelect }) => {
  const [filter, setFilter] = useState('all'); // 'all' | 'rumour' | 'non-rumour'
  const [search, setSearch] = useState('');

  const filtered = useMemo(() => cascades.filter(c => {
    const matchLabel = filter === 'all' || c.label === filter;
    const matchSearch = search === '' || String(c.id).includes(search);
    return matchLabel && matchSearch;
  }), [cascades, filter, search]);

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <h3 style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '1px', color: 'var(--text-secondary)' }}>
          Cascades ({filtered.length}/{cascades.length})
        </h3>
        <div style={{ display: 'flex', gap: '4px' }}>
          {['all', 'rumour', 'non-rumour'].map(f => (
            <button key={f} onClick={() => setFilter(f)} style={{
              fontSize: '0.65rem', padding: '2px 8px', borderRadius: '999px', cursor: 'pointer',
              border: `1px solid ${filter === f ? 'rgba(99,102,241,0.5)' : 'var(--border-color)'}`,
              background: filter === f ? 'rgba(99,102,241,0.15)' : 'transparent',
              color: filter === f ? '#a5b4fc' : 'var(--text-muted)',
            }}>
              {f === 'all' ? 'All' : f === 'rumour' ? 'Fake' : 'Real'}
            </button>
          ))}
        </div>
      </div>

      {cascades.length > 5 && (
        <input
          type="text"
          placeholder="Search cascade ID…"
          value={search}
          onChange={e => setSearch(e.target.value)}
          style={{
            width: '100%', marginBottom: '10px', padding: '7px 12px',
            background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-color)',
            borderRadius: '8px', color: 'white', fontSize: '0.8rem',
            fontFamily: 'inherit', outline: 'none',
          }}
        />
      )}

      {filtered.length === 0 ? (
        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textAlign: 'center', padding: '20px' }}>
          No cascades match.
        </p>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '320px', overflowY: 'auto', paddingRight: '4px' }}>
          {filtered.map(cascade => {
            const isSelected = selectedId === cascade.id;
            const isFake = cascade.label === 'rumour';
            return (
              <div
                key={cascade.id}
                onClick={() => onSelect(cascade.id)}
                style={{
                  cursor: 'pointer', padding: '10px 14px', borderRadius: '10px',
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  background: isSelected ? 'rgba(99,102,241,0.08)' : 'var(--bg-card)',
                  border: `1px solid ${isSelected ? 'rgba(99,102,241,0.4)' : 'var(--border-color)'}`,
                  boxShadow: isSelected ? '0 0 12px rgba(99,102,241,0.08)' : 'none',
                  transition: 'all 0.15s ease',
                }}
              >
                <div>
                  <div style={{ fontWeight: 500, fontSize: '0.875rem', marginBottom: '2px' }}>
                    Cascade #{cascade.id}
                  </div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                    {cascade.nodes}N · {cascade.edges}E
                  </div>
                </div>
                <span style={{
                  fontSize: '0.65rem', padding: '2px 8px', borderRadius: '999px',
                  background: isFake ? 'rgba(239,68,68,0.12)' : 'rgba(99,102,241,0.12)',
                  color: isFake ? '#fca5a5' : '#a5b4fc',
                  border: `1px solid ${isFake ? 'rgba(239,68,68,0.25)' : 'rgba(99,102,241,0.25)'}`,
                  fontWeight: 600,
                }}>
                  {isFake ? 'FAKE' : 'REAL'}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default CascadeList;