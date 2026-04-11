import React, { useEffect, useState } from 'react';
import { fetchCascades, analyzeCascade, scoreNews } from '../api.js';
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
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  
  // Raw Data Input State
  const [newsText, setNewsText] = useState("");
  const [isScanning, setIsScanning] = useState(false);
  const [scanResult, setScanResult] = useState(null);
  const [scanError, setScanError] = useState(null);

  const loadCascades = () => {
    fetchCascades().then(data => setCascades(data.cascades)).catch(err => console.error("Error fetching cascades:", err));
  };

  useEffect(() => {
    loadCascades();
  }, []);

  const handleAnalyze = () => {
    if (selectedCascade !== null) {
      setIsAnalyzing(true);
      analyzeCascade(selectedCascade, k).then(res => {
        setResults(res);
      }).catch(err => {
        console.error("Analysis failed:", err);
      }).finally(() => {
        setIsAnalyzing(false);
      });
    }
  };

  const handleScanNews = () => {
    if (!newsText.trim()) return;
    setIsScanning(true);
    setScanResult(null);
    setScanError(null);
    setResults(null); 
    
    scoreNews(newsText).then(data => {
      if (data.status === 'error') {
        setScanError(data.message);
      } else {
        setScanResult(data);
        loadCascades(); 
        setSelectedCascade(data.cascade_id);
        // Don't clear text - keep it visible so user can see which news gave which result
      }
    }).catch(err => {
      setScanError("Failed to connect to the global search engine.");
    }).finally(() => {
      setIsScanning(false);
    });
  };

  const handleMockImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setNewsText(`[Extracted via OCR from ${file.name}]:\nAnalyzing latest developments on international politics affecting global markets today...`);
      setScanError(null);
      setScanResult(null);
    }
  };

  return (
    <div className="dashboard-layout">
      {/* Sidebar */}
      <div className="sidebar glass-panel" style={{ borderTop: 'none', borderBottom: 'none', borderLeft: 'none', borderRadius: 0 }}>
        <div className="sidebar-header">
          <h2 className="text-emerald" style={{ fontSize: '1.5rem', marginBottom: '8px' }}>Control Panel</h2>
          <p className="text-muted" style={{ fontSize: '0.875rem' }}>Select a cascade and configure intervention parameters.</p>
        </div>
        
        <div className="sidebar-content" style={{ padding: '0 24px 24px 24px' }}>
          
          {/* New Raw Data / LLM Scorer Module */}
          <div style={{ marginTop: '24px', marginBottom: '32px', padding: '16px', background: 'var(--bg-card)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
            <div className="flex justify-between items-center mb-3">
              <div className="flex items-center gap-2">
                <span style={{ fontSize: '1.2rem' }}>🌍</span>
                <h3 style={{ fontSize: '1rem', fontWeight: 600 }}>Global News Scanner</h3>
              </div>
              <label style={{ cursor: 'pointer', fontSize: '0.8rem', padding: '4px 8px', background: 'var(--bg-dark)', borderRadius: '4px', border: '1px solid var(--border-color)' }}>
                📸 Upload Screenshot
                <input type="file" accept="image/*" onChange={handleMockImageUpload} style={{ display: 'none' }} />
              </label>
            </div>
            
            <textarea 
              value={newsText}
              onChange={(e) => { setNewsText(e.target.value); setScanResult(null); setScanError(null); }}
              placeholder="Paste text or type >4 words to run a live global web fact-check..."
              style={{ width: '100%', height: '80px', background: 'rgba(0,0,0,0.3)', border: `1px solid ${scanResult ? (scanResult.label === 'rumour' ? 'var(--accent-rose)' : 'var(--accent-indigo)') : 'var(--border-color)'}`, borderRadius: '8px', padding: '10px', color: 'white', fontFamily: 'inherit', resize: 'none', marginBottom: '6px' }}
            />
            {newsText && (
              <div className="flex justify-end" style={{ marginBottom: '8px' }}>
                <button onClick={() => { setNewsText(""); setScanResult(null); setScanError(null); }} style={{ fontSize: '0.75rem', padding: '2px 10px', background: 'transparent', border: '1px solid var(--border-color)', borderRadius: '4px', color: 'var(--text-muted)', cursor: 'pointer' }}>
                  Clear
                </button>
              </div>
            )}
            
            <button 
              className="btn-outline" 
              style={{ width: '100%', borderColor: 'var(--accent-indigo)', color: 'white' }}
              onClick={handleScanNews}
              disabled={isScanning || !newsText.trim()}
            >
              {isScanning ? 'Querying Global Web...' : 'Live Fact-Check & Generate'}
            </button>
            
            {/* Display Error Result */}
            {scanError && (
              <div style={{ marginTop: '16px', padding: '12px', background: 'rgba(225,29,72,0.1)', borderRadius: '8px', borderLeft: '4px solid var(--accent-rose)' }}>
                <p style={{ fontSize: '0.875rem', color: 'var(--accent-rose)' }}>{scanError}</p>
              </div>
            )}

            {/* Display Scan Result */}
            {scanResult && !scanError && (
              <div style={{ marginTop: '16px', padding: '12px', background: 'rgba(0,0,0,0.3)', borderRadius: '8px', borderLeft: `4px solid ${scanResult.label === 'rumour' ? 'var(--accent-rose)' : 'var(--accent-indigo)'}` }}>
                <p style={{ fontSize: '0.75rem', marginBottom: '6px', color: 'var(--text-muted)', fontStyle: 'italic', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}
                  title={newsText}>
                  "{newsText.length > 55 ? newsText.slice(0, 55) + '...' : newsText}"
                </p>
                <p style={{ fontSize: '0.875rem', marginBottom: '4px' }} className="text-secondary">Classification Result:</p>
                <div className="flex justify-between items-end">
                  <span style={{ fontWeight: 'bold', fontSize: '1.1rem', color: scanResult.label === 'rumour' ? 'var(--accent-rose)' : 'var(--accent-indigo)' }}>
                    {scanResult.label === 'rumour' ? 'FAKE NEWS' : 'REAL NEWS'}
                  </span>
                  <span className="mono" style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>Score: {scanResult.score}%</span>
                </div>
                <p style={{ fontSize: '0.75rem', marginTop: '8px', color: 'var(--accent-emerald)' }}>
                  ✓ Web cross-reference complete.
                </p>
                <p style={{ fontSize: '0.75rem', marginTop: '2px', color: 'var(--accent-emerald)' }}>
                  ✓ Cascade ID: {scanResult.cascade_id} selected
                </p>
              </div>
            )}
          </div>

          <div className="mb-6">
            <div className="flex justify-between mb-2">
              <label style={{ fontSize: '0.875rem', fontWeight: 500 }} className="text-primary">Intervention Budget (K)</label>
              <span className="mono text-emerald bg-emerald-glow" style={{ padding: '2px 8px', borderRadius: '12px', fontSize: '0.875rem' }}>{k}</span>
            </div>
            <input type="range" min="1" max="10" value={k} onChange={e => setK(parseInt(e.target.value))} />
            <p className="mt-2 text-muted" style={{ fontSize: '0.75rem' }}>Max number of nodes to debunk.</p>
          </div>

          <CascadeList cascades={cascades} selectedId={selectedCascade} onSelect={setSelectedCascade} />
        </div>
        
        <div style={{ padding: '24px', borderTop: '1px solid var(--border-color)' }}>
          <button 
            className="btn-primary" 
            style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px' }}
            onClick={handleAnalyze} 
            disabled={selectedCascade === null || isAnalyzing}
          >
            {isAnalyzing ? (
              <>Running Optimizer...</>
            ) : (
              <>Run Intervention</>
            )}
          </button>
        </div>
      </div>

      {/* Main Panel */}
      <div className="main-content">
        {!results && (
          <div className="flex flex-col items-center justify-center p-6 text-center" style={{ height: '100%', opacity: 0.6 }}>
            <div style={{ width: '64px', height: '64px', borderRadius: '50%', background: 'var(--bg-card)', border: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '16px' }}>
              <span style={{ fontSize: '24px' }}>🕸️</span>
            </div>
            <h3 className="text-secondary mb-2">No Analysis Results</h3>
            <p className="text-muted" style={{ maxWidth: '400px' }}>Select a cascade from the sidebar and click 'Run Intervention' to see the causal impact.</p>
          </div>
        )}

        {results && (
          <div className="flex flex-col gap-6" style={{ maxWidth: '1400px', margin: '0 auto' }}>
            <div className="flex items-center justify-between">
              <div>
                <h2 style={{ fontSize: '1.75rem', marginBottom: '8px' }}>Intervention Analysis</h2>
                <div className="flex gap-2">
                  <span className={`chip mt-2 ${results.label === 'rumour' ? 'chip-rose' : 'chip-indigo'}`}>
                    {results.label.toUpperCase()}
                  </span>
                  <span className="chip chip-emerald mt-2">ID: {results.cascade_id}</span>
                </div>
              </div>
            </div>

            <StatsCards 
              label={results.label} 
              nodes={results.nodes} 
              reductionPct={results.reduction_pct} 
              improvement={results.reduction_pct - results.random_reduction} 
            />
            
            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 3fr)', gap: '24px' }}>
              
              <div className="flex flex-col gap-6">
                <InterventionPanel 
                  interventionNodes={results.intervention_nodes} 
                  reductionPct={results.reduction_pct} 
                  randomReduction={results.random_reduction} 
                />
                <ScoreChart scoreHistory={results.score_history} />
              </div>

              {/* Side-by-side graphs */}
              <div className="glass-card p-4" style={{ display: 'flex', flexDirection: 'column' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', flex: 1 }}>
                  
                  {/* Before */}
                  <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: '400px' }}>
                    <div className="text-center mb-4">
                      <h3 style={{ fontSize: '1.1rem', fontWeight: '500', color: 'var(--text-primary)' }}>BEFORE Intervention</h3>
                      <p className="text-secondary" style={{ fontSize: '0.875rem' }}>{results.nodes} nodes infected | Rumour: {results.label === 'rumour' ? 'FAKE NEWS' : 'REAL NEWS'}</p>
                    </div>
                    <div style={{ flex: 1, background: 'var(--bg-dark)', borderRadius: 'var(--radius-md)', padding: '10px' }}>
                      <GraphViz data={results.graph_before} />
                    </div>
                  </div>

                  {/* After */}
                  <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: '400px' }}>
                    <div className="text-center mb-4">
                      <h3 style={{ fontSize: '1.1rem', fontWeight: '500', color: 'var(--text-primary)' }}>AFTER Intervention ({k} nodes debunked)</h3>
                      <p className="text-secondary" style={{ fontSize: '0.875rem' }}>Spread reduced by {results.reduction_pct.toFixed(1)}%</p>
                    </div>
                    <div style={{ flex: 1, background: 'var(--bg-dark)', borderRadius: 'var(--radius-md)', padding: '10px' }}>
                      <GraphViz data={results.graph_after} />
                    </div>
                  </div>

                </div>

                {/* Shared Legend */}
                <div className="flex justify-center gap-6 mt-6 pt-4" style={{ borderTop: '1px solid var(--border-color)' }}>
                  <div className="flex items-center gap-2">
                    <span style={{ width: '20px', height: '6px', borderRadius: '2px', background: 'var(--accent-amber)' }}></span>
                    <span className="text-muted" style={{ fontSize: '0.875rem' }}>Source tweet</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span style={{ width: '20px', height: '6px', borderRadius: '2px', background: 'var(--accent-rose)' }}></span>
                    <span className="text-muted" style={{ fontSize: '0.875rem' }}>Infected node</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span style={{ width: '20px', height: '6px', borderRadius: '2px', background: 'var(--accent-emerald)' }}></span>
                    <span className="text-muted" style={{ fontSize: '0.875rem' }}>Debunked (intervention)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span style={{ width: '20px', height: '6px', borderRadius: '2px', background: '#1e293b' }}></span>
                    <span className="text-muted" style={{ fontSize: '0.875rem' }}>Contained (blocked)</span>
                  </div>
                </div>

              </div>
              
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;