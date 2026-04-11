import React, { useEffect, useState, useCallback } from 'react';
import { fetchCascades, analyzeCascade, scoreNews, fetchHealth, fetchWorldHeadlines } from '../api.js';
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
  const [analyzeError, setAnalyzeError] = useState(null);
  const [backendStatus, setBackendStatus] = useState(null); // 'ok' | 'error'

  const [newsText, setNewsText] = useState('');
  const [isScanning, setIsScanning] = useState(false);
  const [scanResult, setScanResult] = useState(null);
  const [scanError, setScanError] = useState(null);

  const [worldHeadlines, setWorldHeadlines] = useState([]);
  const [worldLoading, setWorldLoading] = useState(false);
  const [worldError, setWorldError] = useState(null);
  const [worldProvider, setWorldProvider] = useState(null);

  const loadWorldHeadlines = useCallback(() => {
    setWorldLoading(true);
    setWorldError(null);
    fetchWorldHeadlines(28)
      .then((data) => {
        setWorldHeadlines(data.articles || []);
        setWorldProvider(data.provider || null);
      })
      .catch((err) => {
        setWorldHeadlines([]);
        setWorldProvider(null);
        setWorldError(err.message || 'Could not load headlines');
      })
      .finally(() => setWorldLoading(false));
  }, []);

  // Check backend health on mount
  useEffect(() => {
    fetchHealth()
      .then(d => setBackendStatus(d.status === 'ok' ? 'ok' : 'error'))
      .catch(() => setBackendStatus('error'));
  }, []);

  useEffect(() => {
    if (backendStatus === 'ok') loadWorldHeadlines();
  }, [backendStatus, loadWorldHeadlines]);

  const loadCascades = useCallback(() => {
    fetchCascades()
      .then(data => setCascades(data.cascades || []))
      .catch(err => console.error('Error fetching cascades:', err));
  }, []);

  useEffect(() => { loadCascades(); }, [loadCascades]);

  const handleAnalyze = async () => {
    if (selectedCascade === null) return;
    setIsAnalyzing(true);
    setAnalyzeError(null);
    try {
      const res = await analyzeCascade(selectedCascade, k);
      setResults(res);
    } catch (err) {
      setAnalyzeError(err.message || 'Analysis failed. Is the backend running?');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleScanNews = async () => {
    if (!newsText.trim()) return;
    setIsScanning(true);
    setScanResult(null);
    setScanError(null);
    setResults(null);
    try {
      const data = await scoreNews(newsText);
      if (data.status === 'error') {
        setScanError(data.message);
      } else {
        setScanResult(data);
        loadCascades();
        setSelectedCascade(data.cascade_id);
      }
    } catch (err) {
      setScanError(err.message || 'Failed to connect to backend.');
    } finally {
      setIsScanning(false);
    }
  };

  const handleMockImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setNewsText(`[Extracted via OCR from ${file.name}]:\nAnalyzing latest developments on international politics affecting global markets today...`);
      setScanError(null);
      setScanResult(null);
    }
  };

  const clearScan = () => {
    setNewsText('');
    setScanResult(null);
    setScanError(null);
  };

  const isFake = scanResult?.label === 'rumour';

  return (
    <div className="dashboard-layout">

      {/* ── Sidebar ── */}
      <div className="sidebar glass-panel" style={{ borderTop: 'none', borderBottom: 'none', borderLeft: 'none', borderRadius: 0 }}>

        <div className="sidebar-header">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
            <h2 className="text-emerald" style={{ fontSize: '1.4rem' }}>Control Panel</h2>
            {backendStatus && (
              <span style={{
                fontSize: '0.7rem', padding: '2px 8px', borderRadius: '999px',
                background: backendStatus === 'ok' ? 'rgba(16,185,129,0.15)' : 'rgba(239,68,68,0.15)',
                color: backendStatus === 'ok' ? 'var(--accent-emerald)' : 'var(--accent-rose)',
                border: `1px solid ${backendStatus === 'ok' ? 'rgba(16,185,129,0.3)' : 'rgba(239,68,68,0.3)'}`,
              }}>
                {backendStatus === 'ok' ? '● Live' : '● Offline'}
              </span>
            )}
          </div>
          <p className="text-muted" style={{ fontSize: '0.8rem' }}>
            Classify news, select a cascade, and run causal intervention.
          </p>
        </div>

        <div className="sidebar-content">

          {/* ── Global News Scanner ── */}
          <div style={{ marginBottom: '28px', padding: '16px', background: 'var(--bg-card)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span>🌍</span>
                <h3 style={{ fontSize: '0.95rem', fontWeight: 600 }}>Global News Scanner</h3>
              </div>
              <label style={{ cursor: 'pointer', fontSize: '0.75rem', padding: '4px 10px', background: 'var(--bg-dark)', borderRadius: '6px', border: '1px solid var(--border-color)', color: 'var(--text-muted)' }}>
                📸 Upload
                <input type="file" accept="image/*" onChange={handleMockImageUpload} style={{ display: 'none' }} />
              </label>
            </div>

            <textarea
              value={newsText}
              onChange={e => { setNewsText(e.target.value); setScanResult(null); setScanError(null); }}
              placeholder="Paste a news headline or claim to fact-check in real time..."
              rows={3}
              style={{
                width: '100%', background: 'rgba(0,0,0,0.3)', resize: 'vertical',
                border: `1px solid ${scanResult ? (isFake ? 'var(--accent-rose)' : 'var(--accent-emerald)') : 'var(--border-color)'}`,
                borderRadius: '8px', padding: '10px', color: 'white',
                fontFamily: 'inherit', fontSize: '0.875rem', lineHeight: 1.5,
                marginBottom: '8px', outline: 'none',
              }}
            />

            <div style={{ display: 'flex', gap: '8px', marginBottom: '10px' }}>
              <button
                className="btn-outline"
                style={{ flex: 1, borderColor: 'var(--accent-indigo)', color: 'white', padding: '8px 12px', fontSize: '0.85rem' }}
                onClick={handleScanNews}
                disabled={isScanning || !newsText.trim()}
              >
                {isScanning ? '⏳ Checking...' : '🔍 Fact-Check Live'}
              </button>
              {newsText && (
                <button onClick={clearScan} className="btn-outline" style={{ padding: '8px 12px', fontSize: '0.8rem' }}>
                  ✕
                </button>
              )}
            </div>

            {/* Live world headlines — click to fact-check */}
            <div style={{ marginBottom: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                  Live headlines (multi-region)
                </span>
                <button
                  type="button"
                  onClick={loadWorldHeadlines}
                  disabled={worldLoading || backendStatus !== 'ok'}
                  className="btn-outline"
                  style={{ padding: '4px 10px', fontSize: '0.7rem', opacity: backendStatus !== 'ok' ? 0.5 : 1 }}
                >
                  {worldLoading ? '…' : '↻ Refresh'}
                </button>
              </div>
              {worldError && (
                <p style={{ fontSize: '0.72rem', color: 'var(--accent-rose)', marginBottom: '6px' }}>{worldError}</p>
              )}
              {!worldLoading && worldProvider && (
                <p style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginBottom: '6px' }}>
                  Source mix: {worldProvider.replace(/\+/g, ' + ')}
                </p>
              )}
              <div
                style={{
                  maxHeight: '200px',
                  overflowY: 'auto',
                  borderRadius: '8px',
                  border: '1px solid var(--border-color)',
                  background: 'rgba(0,0,0,0.25)',
                }}
              >
                {worldLoading && worldHeadlines.length === 0 && (
                  <p style={{ padding: '12px', fontSize: '0.75rem', color: 'var(--text-muted)' }}>Loading headlines…</p>
                )}
                {!worldLoading && worldHeadlines.length === 0 && !worldError && backendStatus === 'ok' && (
                  <p style={{ padding: '12px', fontSize: '0.75rem', color: 'var(--text-muted)' }}>No headlines returned. Try Refresh.</p>
                )}
                {worldHeadlines.map((a, idx) => (
                  <button
                    key={`${a.title}-${idx}`}
                    type="button"
                    onClick={() => {
                      const clean = (a.title || '').replace(/\s*[-–—]\s*[^-–—]+$/, '').trim() || a.title;
                      setNewsText(clean);
                      setScanResult(null);
                      setScanError(null);
                    }}
                    style={{
                      display: 'block',
                      width: '100%',
                      textAlign: 'left',
                      padding: '8px 10px',
                      fontSize: '0.72rem',
                      lineHeight: 1.35,
                      color: 'var(--text-secondary)',
                      background: 'transparent',
                      border: 'none',
                      borderBottom: '1px solid var(--border-color)',
                      cursor: 'pointer',
                    }}
                  >
                    <span style={{ color: 'var(--accent-indigo)', marginRight: '6px', fontSize: '0.65rem' }}>{a.tag || '·'}</span>
                    {a.title}
                    <span style={{ display: 'block', fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                      {a.source || 'News'}
                    </span>
                  </button>
                ))}
              </div>
            </div>

            {/* Error */}
            {scanError && (
              <div style={{ padding: '10px 12px', background: 'rgba(239,68,68,0.1)', borderRadius: '8px', borderLeft: '3px solid var(--accent-rose)', marginBottom: '8px' }}>
                <p style={{ fontSize: '0.8rem', color: 'var(--accent-rose)' }}>{scanError}</p>
              </div>
            )}

            {/* Result */}
            {scanResult && !scanError && (
              <div style={{
                padding: '12px', borderRadius: '10px',
                background: isFake ? 'rgba(239,68,68,0.08)' : 'rgba(16,185,129,0.08)',
                border: `1px solid ${isFake ? 'rgba(239,68,68,0.3)' : 'rgba(16,185,129,0.3)'}`,
              }}>
                <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: '8px', fontStyle: 'italic', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={newsText}>
                  "{newsText.length > 55 ? newsText.slice(0, 55) + '…' : newsText}"
                </p>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontWeight: 700, fontSize: '1.1rem', color: isFake ? 'var(--accent-rose)' : 'var(--accent-emerald)' }}>
                    {isFake ? '🔴 FAKE NEWS' : '🟢 REAL NEWS'}
                  </span>
                  <span className="mono" style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    {scanResult.score}%
                  </span>
                </div>
                <div style={{ marginTop: '8px', fontSize: '0.72rem', color: 'var(--text-muted)', display: 'flex', flexWrap: 'wrap', gap: '12px' }}>
                  <span>✓ Method: {scanResult.method || 'ml+rules'}</span>
                  <span>✓ Cascade #{scanResult.cascade_id}</span>
                </div>
                {Array.isArray(scanResult.sources) && scanResult.sources.length > 0 && (
                  <p style={{ marginTop: '8px', fontSize: '0.68rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
                    Outlets: {scanResult.sources.slice(0, 5).join(', ')}
                  </p>
                )}
              </div>
            )}
          </div>

          {/* ── Intervention Budget ── */}
          <div style={{ marginBottom: '28px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <label style={{ fontSize: '0.875rem', fontWeight: 500 }}>Intervention Budget (K)</label>
              <span className="mono text-emerald" style={{ background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.3)', padding: '2px 10px', borderRadius: '999px', fontSize: '0.875rem' }}>
                {k}
              </span>
            </div>
            <input type="range" min="1" max="10" value={k} onChange={e => setK(parseInt(e.target.value))} />
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '6px' }}>
              Max nodes the optimizer is allowed to suppress.
            </p>
          </div>

          {/* ── Cascade List ── */}
          <CascadeList cascades={cascades} selectedId={selectedCascade} onSelect={setSelectedCascade} />
        </div>

        <div style={{ padding: '20px 24px', borderTop: '1px solid var(--border-color)' }}>
          <button
            className="btn-primary"
            style={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px', padding: '12px' }}
            onClick={handleAnalyze}
            disabled={selectedCascade === null || isAnalyzing}
          >
            {isAnalyzing ? '⏳ Running Optimizer…' : '🎯 Run Intervention'}
          </button>
          {analyzeError && (
            <p style={{ fontSize: '0.75rem', color: 'var(--accent-rose)', marginTop: '8px', textAlign: 'center' }}>
              {analyzeError}
            </p>
          )}
        </div>
      </div>

      {/* ── Main Content ── */}
      <div className="main-content">

        {/* Empty state */}
        {!results && !isAnalyzing && (
          <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', opacity: 0.5 }}>
            <div style={{ fontSize: '4rem', marginBottom: '16px' }}>🕸️</div>
            <h3 style={{ marginBottom: '8px', color: 'var(--text-secondary)' }}>No Analysis Yet</h3>
            <p style={{ color: 'var(--text-muted)', maxWidth: '360px', textAlign: 'center', lineHeight: 1.6 }}>
              Paste a news headline in the scanner, or select a cascade from the sidebar, then click <strong>Run Intervention</strong>.
            </p>
          </div>
        )}

        {/* Loading state */}
        {isAnalyzing && (
          <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '16px' }}>
            <div style={{ width: '48px', height: '48px', border: '3px solid var(--border-color)', borderTop: '3px solid var(--accent-emerald)', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
            <p style={{ color: 'var(--text-secondary)' }}>Running causal intervention optimizer…</p>
          </div>
        )}

        {/* Results */}
        {results && !isAnalyzing && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px', maxWidth: '1400px', margin: '0 auto' }}>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '12px' }}>
              <div>
                <h2 style={{ fontSize: '1.6rem', marginBottom: '8px', fontWeight: 700 }}>Intervention Analysis</h2>
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                  <span className={`chip ${results.label === 'rumour' ? 'chip-rose' : 'chip-indigo'}`}>
                    {results.label === 'rumour' ? '🔴 FAKE NEWS' : '🟢 REAL NEWS'}
                  </span>
                  <span className="chip chip-emerald">Cascade #{results.cascade_id}</span>
                  <span className="chip chip-amber">GAT score: {(results.graph_fake_prob * 100).toFixed(1)}%</span>
                </div>
              </div>
              <button className="btn-outline" onClick={() => setResults(null)} style={{ fontSize: '0.8rem', padding: '6px 14px' }}>
                ✕ Clear
              </button>
            </div>

            <StatsCards
              label={results.label}
              nodes={results.nodes}
              reductionPct={results.reduction_pct}
              improvement={results.reduction_pct - results.random_reduction}
              graphFakeProb={results.graph_fake_prob}
            />

            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1fr) minmax(0,3fr)', gap: '20px' }}>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <InterventionPanel
                  interventionNodes={results.intervention_nodes}
                  reductionPct={results.reduction_pct}
                  randomReduction={results.random_reduction}
                  baselineScore={results.baseline_score}
                  finalScore={results.final_score}
                />
                <ScoreChart scoreHistory={results.score_history} />
              </div>

              {/* Graph panel */}
              <div className="glass-card" style={{ padding: '20px', position: 'sticky', top: '20px', alignSelf: 'start' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', height: '100%' }}>

                  <div style={{ display: 'flex', flexDirection: 'column', minHeight: '420px' }}>
                    <div style={{ textAlign: 'center', marginBottom: '12px' }}>
                      <h3 style={{ fontSize: '1rem', fontWeight: 600 }}>BEFORE Intervention</h3>
                      <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                        {results.nodes} nodes · Baseline score: {results.baseline_score.toFixed(3)}
                      </p>
                    </div>
                    <div style={{ height: '460px', background: 'var(--bg-dark)', borderRadius: '10px', padding: '8px', overflow: 'hidden' }}>
                      <GraphViz data={results.graph_before} />
                    </div>
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', minHeight: '420px' }}>
                    <div style={{ textAlign: 'center', marginBottom: '12px' }}>
                      <h3 style={{ fontSize: '1rem', fontWeight: 600 }}>AFTER Intervention ({k} nodes)</h3>
                      <p style={{ fontSize: '0.8rem', color: 'var(--accent-emerald)' }}>
                        Spread reduced {results.reduction_pct.toFixed(1)}%
                      </p>
                    </div>
                    <div style={{ height: '460px', background: 'var(--bg-dark)', borderRadius: '10px', padding: '8px', overflow: 'hidden' }}>
                      <GraphViz data={results.graph_after} />
                    </div>
                  </div>

                </div>

                {/* Legend */}
                <div style={{ display: 'flex', justifyContent: 'center', gap: '24px', marginTop: '16px', paddingTop: '16px', borderTop: '1px solid var(--border-color)', flexWrap: 'wrap' }}>
                  {[
                    { color: 'var(--accent-amber)', label: 'Source node' },
                    { color: 'var(--accent-rose)', label: 'Infected' },
                    { color: 'var(--accent-emerald)', label: 'Debunked' },
                    { color: '#1e293b', label: 'Contained' },
                  ].map(({ color, label }) => (
                    <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span style={{ width: '20px', height: '5px', borderRadius: '2px', background: color, display: 'inline-block' }} />
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{label}</span>
                    </div>
                  ))}
                </div>
              </div>

            </div>
          </div>
        )}
      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
};

export default Dashboard;
