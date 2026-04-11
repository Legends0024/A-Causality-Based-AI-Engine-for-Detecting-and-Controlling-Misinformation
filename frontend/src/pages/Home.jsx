import React from 'react';
import { useNavigate } from 'react-router-dom';

const ArchitectureFlow = () => {
  const makeNode = (title, subtitle, color, borderColor) => (
    <div style={{
      background: color,
      border: `1px solid ${borderColor}`,
      borderRadius: '8px',
      padding: '16px',
      width: '200px',
      textAlign: 'center',
      boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
      zIndex: 2,
      position: 'relative'
    }}>
      <div style={{ fontWeight: 600, fontSize: '1rem', color: '#fff', marginBottom: '4px' }}>{title}</div>
      <div style={{ fontSize: '0.8rem', color: 'rgba(255,255,255,0.7)' }}>{subtitle}</div>
    </div>
  );

  return (
    <div style={{ marginTop: '60px', padding: '40px', background: 'rgba(16, 23, 42, 0.4)', borderRadius: '16px', border: '1px solid var(--border-color)', position: 'relative' }}>
      <h3 style={{ textAlign: 'center', marginBottom: '40px', color: 'var(--text-secondary)' }}>System Architecture Pipeline</h3>
      
      {/* SVG for connecting lines */}
      <svg style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 1 }}>
        <defs>
          <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="rgba(255,255,255,0.3)" />
          </marker>
        </defs>
        {/* Top row arrows */}
        <line x1="220" y1="135" x2="260" y2="135" stroke="rgba(255,255,255,0.3)" strokeWidth="2" markerEnd="url(#arrowhead)" />
        <line x1="480" y1="135" x2="520" y2="135" stroke="rgba(255,255,255,0.3)" strokeWidth="2" markerEnd="url(#arrowhead)" />
        <line x1="740" y1="135" x2="780" y2="135" stroke="rgba(255,255,255,0.3)" strokeWidth="2" markerEnd="url(#arrowhead)" />
        
        {/* Downward arrow from Causal Graph to Intervention */}
        <line x1="880" y1="175" x2="880" y2="235" stroke="rgba(255,255,255,0.3)" strokeWidth="2" markerEnd="url(#arrowhead)" />
        
        {/* Bottom row arrow (Right to Left) */}
        <line x1="780" y1="275" x2="685" y2="275" stroke="rgba(255,255,255,0.3)" strokeWidth="2" markerEnd="url(#arrowhead)" />
        
        {/* Dotted line from Temporal GNN to Output Layer */}
        <line x1="380" y1="175" x2="580" y2="245" stroke="rgba(255,255,255,0.3)" strokeWidth="2" strokeDasharray="5,5" markerEnd="url(#arrowhead)" />
      </svg>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '60px', alignItems: 'center' }}>
        {/* Top Row */}
        <div style={{ display: 'flex', gap: '60px', justifyContent: 'center' }}>
          {makeNode("Raw data", "Twitter / Reddit posts", "#2d2d2d", "#4d4d4d")}
          {makeNode("Temporal GNN", "Models spread over time", "#3b3086", "#5044a8")}
          {makeNode("LLM scorer", "Content credibility score", "#0e5239", "#177e58")}
          {makeNode("Causal graph", "do-calculus layer", "#854d10", "#b56916")}
        </div>
        
        {/* Bottom Row */}
        <div style={{ display: 'flex', gap: '60px', justifyContent: 'flex-end', width: '100%', paddingRight: '20px' }}>
          {makeNode("Output layer", "Dashboard + alerts", "#265410", "#367517")}
          {makeNode("Intervention optimizer", "Find minimum debunk node", "#6b261b", "#963626")}
        </div>
      </div>
    </div>
  );
};

const Home = () => {
  const navigate = useNavigate();

  return (
    <div className="flex flex-col items-center p-6" style={{ minHeight: '100vh', position: 'relative' }}>
      
      {/* Background glowing orb */}
      <div style={{
        position: 'absolute',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        width: '60vw',
        height: '60vw',
        background: 'radial-gradient(circle, rgba(16,185,129,0.05) 0%, rgba(7,9,14,0) 70%)',
        zIndex: -1,
        pointerEvents: 'none'
      }}></div>

      <div style={{ maxWidth: '1000px', width: '100%', margin: '0 auto', padding: '40px 0' }}>
        <div className="flex flex-col items-center text-center">
          <div className="flex justify-center gap-4 mb-6">
            <span className="chip chip-indigo">GNN + LLM fusion</span>
            <span className="chip chip-emerald">Causal AI</span>
            <span className="chip chip-amber">Temporally dynamic</span>
          </div>
          
          <h1 style={{ fontSize: '3.5rem', lineHeight: '1.2', marginBottom: '24px', fontWeight: 700 }} className="text-primary">
            Causal intervention engine for <span className="text-emerald">misinformation containment</span>
          </h1>
          
          <p className="text-secondary" style={{ fontSize: '1.25rem', lineHeight: '1.6', maxWidth: '800px', marginBottom: '40px' }}>
            Don't just predict where fake news spreads — ask 'if we debunk node X at time T, what's the counterfactual spread?' Build an AI that runs causal interventions on live social graphs to find the minimum set of nodes to target that collapses a rumour's reach.
          </p>

          <div className="flex gap-4">
            <button 
              className="btn-primary" 
              style={{ fontSize: '1.1rem', padding: '14px 32px' }}
              onClick={() => navigate('/dashboard')}
            >
              Launch Engine Dashboard →
            </button>
            <a 
              href="https://github.com" 
              target="_blank" 
              rel="noopener noreferrer" 
              className="btn-outline flex items-center justify-center" 
              style={{ padding: '14px 32px' }}
            >
              GitHub Repository
            </a>
          </div>
        </div>

        {/* System Architecture Section */}
        <ArchitectureFlow />
        
      </div>
    </div>
  );
};

export default Home;