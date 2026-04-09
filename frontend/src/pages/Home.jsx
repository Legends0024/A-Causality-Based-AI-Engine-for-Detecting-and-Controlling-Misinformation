import React, { useEffect, useState } from 'react';
import { fetchStats } from '../api.js';

const Home = () => {
  const [stats, setStats] = useState(null);
  useEffect(() => {
    fetchStats().then(setStats);
  }, []);
  return (
    <div style={{ background: '#020817', color: 'white', minHeight: '100vh', fontFamily: 'IBM Plex Mono', padding: '20px' }}>
      <h1 style={{ textAlign: 'center', fontSize: '3em' }}>Causal Intervention Engine</h1>
      <p style={{ textAlign: 'center', fontSize: '1.2em' }}>Find the minimum nodes to debunk to collapse misinformation spread</p>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px', margin: '40px 0' }}>
        <div style={{ background: '#0a0f1e', padding: '20px', border: '1px solid #1e293b', borderRadius: '10px' }}>
          <h3>GAT Model</h3>
          <p>Graph Attention Network predicts spread probability for each node.</p>
        </div>
        <div style={{ background: '#0a0f1e', padding: '20px', border: '1px solid #1e293b', borderRadius: '10px' }}>
          <h3>Greedy Optimizer</h3>
          <p>Finds the K nodes whose debunking minimizes total spread score.</p>
        </div>
        <div style={{ background: '#0a0f1e', padding: '20px', border: '1px solid #1e293b', borderRadius: '10px' }}>
          <h3>Before/After</h3>
          <p>Visualize the graph before and after intervention.</p>
        </div>
      </div>
      {stats && (
        <div style={{ textAlign: 'center' }}>
          <h2>Dataset Stats</h2>
          <p>Total Cascades: {stats.total_cascades}</p>
          <p>Label Distribution: {Object.entries(stats.label_distribution).map(([k,v]) => `${k}: ${v}`).join(', ')}</p>
          <p>Avg Nodes: {stats.avg_nodes.toFixed(1)}, Avg Edges: {stats.avg_edges.toFixed(1)}</p>
        </div>
      )}
      <div style={{ textAlign: 'center', marginTop: '40px' }}>
        <button onClick={() => window.location.href = '/dashboard'} style={{ background: '#10b981', color: 'white', padding: '10px 20px', border: 'none', borderRadius: '5px', cursor: 'pointer' }}>
          Launch Dashboard
        </button>
      </div>
    </div>
  );
};

export default Home;