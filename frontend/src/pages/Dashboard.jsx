import React, { useEffect, useState } from 'react';
import { fetchCascades, analyzeCascade } from '../api.js';
import CascadeList from '../components/CascadeList.jsx';
import StatsCards from '../components/StatsCards.jsx';
import ScoreChart from '../components/ScoreChart.jsx';
import InterventionPanel from '../components/InterventionPanel.jsx';
import GraphViz from '../components/GraphViz.jsx';

const Dashboard = () => {
  const [cascades, setCascades] = useState([]);
  const [selectedCascade, setSelectedCascade] = useState(null);
  const [k, setK] = useState(5);
  const [results, setResults] = useState(null);
  const [graphView, setGraphView] = useState('before');
  useEffect(() => {
    fetchCascades().then(data => setCascades(data.cascades));
  }, []);
  const handleAnalyze = () => {
    if (selectedCascade !== null) {
      analyzeCascade(selectedCascade, k).then(setResults);
    }
  };
  return (
    <div style={{ background: '#020817', color: 'white', minHeight: '100vh', fontFamily: 'IBM Plex Mono', display: 'flex' }}>
      <div style={{ width: '280px', padding: '20px', borderRight: '1px solid #1e293b' }}>
        <h2>Dashboard</h2>
        <div>
          <label>K: {k}</label>
          <input type="range" min="1" max="10" value={k} onChange={e => setK(parseInt(e.target.value))} />
        </div>
        <CascadeList cascades={cascades} selectedId={selectedCascade} onSelect={setSelectedCascade} />
      </div>
      <div style={{ flex: 1, padding: '20px' }}>
        <button onClick={handleAnalyze} disabled={selectedCascade === null} style={{ background: '#10b981', color: 'white', padding: '10px', border: 'none', borderRadius: '5px', cursor: selectedCascade ? 'pointer' : 'not-allowed' }}>
          Run Intervention
        </button>
        {results && (
          <>
            <StatsCards label={results.label} nodes={results.nodes} reductionPct={results.reduction_pct} improvement={results.reduction_pct - results.random_reduction} />
            <InterventionPanel interventionNodes={results.intervention_nodes} reductionPct={results.reduction_pct} randomReduction={results.random_reduction} />
            <ScoreChart scoreHistory={results.score_history} />
            <div>
              <button onClick={() => setGraphView('before')} style={{ marginRight: '10px', background: graphView === 'before' ? '#10b981' : '#0a0f1e', color: 'white', padding: '5px', border: '1px solid #1e293b' }}>Before</button>
              <button onClick={() => setGraphView('after')} style={{ background: graphView === 'after' ? '#10b981' : '#0a0f1e', color: 'white', padding: '5px', border: '1px solid #1e293b' }}>After</button>
            </div>
            <GraphViz data={graphView === 'before' ? results.graph_before : results.graph_after} title={graphView === 'before' ? 'Before Intervention' : 'After Intervention'} />
          </>
        )}
      </div>
    </div>
  );
};

export default Dashboard;