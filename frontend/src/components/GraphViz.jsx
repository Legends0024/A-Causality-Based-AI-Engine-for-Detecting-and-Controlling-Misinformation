import React, { useRef, useEffect, useState, useCallback } from 'react';

const NODE_COLORS = {
  root: '#f59e0b',
  debunked: '#10b981',
  infected: '#ef4444',
  contained: '#1e293b',
};

const NODE_RADIUS = { root: 14, debunked: 10, infected: 7 };

function computeStableLayout(nodes, width, height) {
  if (!nodes || nodes.length === 0) return {};

  const safeWidth = Math.max(width || 0, 320);
  const safeHeight = Math.max(height || 0, 280);
  const paddingX = 36;
  const paddingTop = 34;
  const paddingBottom = 34;

  const maxKnownDepth = Math.max(...nodes.map(n => (n.depth >= 0 ? n.depth : 0)), 0);
  const grouped = {};

  nodes.forEach(node => {
    const effectiveDepth = node.depth >= 0 ? node.depth : maxKnownDepth + 1;
    if (!grouped[effectiveDepth]) grouped[effectiveDepth] = [];
    grouped[effectiveDepth].push(node);
  });

  const orderedDepths = Object.keys(grouped)
    .map(Number)
    .sort((a, b) => a - b);

  const usableHeight = safeHeight - paddingTop - paddingBottom;
  const layerStep = orderedDepths.length > 1
    ? usableHeight / (orderedDepths.length - 1)
    : 0;

  const positions = {};

  orderedDepths.forEach((depth, layerIndex) => {
    const levelNodes = grouped[depth].slice().sort((a, b) => {
      if (a.type === 'root') return -1;
      if (b.type === 'root') return 1;
      if (b.degree !== a.degree) return b.degree - a.degree;
      return String(a.id).localeCompare(String(b.id), undefined, { numeric: true });
    });

    const y = paddingTop + (layerIndex * layerStep);
    const usableWidth = safeWidth - (paddingX * 2);

    levelNodes.forEach((node, index) => {
      const x = levelNodes.length === 1
        ? safeWidth / 2
        : paddingX + ((index + 1) * usableWidth) / (levelNodes.length + 1);
      positions[node.id] = { x, y };
    });
  });

  return positions;
}

const GraphViz = ({ data }) => {
  const containerRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [tooltip, setTooltip] = useState(null);
  const [hoveredNode, setHoveredNode] = useState(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const obs = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect;
      setDimensions({ width, height });
    });
    obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, []);

  const { nodes, edges } = data || { nodes: [], edges: [] };
  const safeWidth = Math.max(dimensions.width || 0, 320);
  const safeHeight = Math.max(dimensions.height || 0, 280);
  const positions = computeStableLayout(nodes, safeWidth, safeHeight);

  const getRadius = (node) => {
    const base = NODE_RADIUS[node.type] || 7;
    return hoveredNode === node.id ? base + 3 : base;
  };

  const getColor = (node) => {
    if (node.type === 'infected' && node.depth === -1) return '#1e293b';
    return NODE_COLORS[node.type] || NODE_COLORS.infected;
  };

  const handleMouseEnter = useCallback((node, e) => {
    setHoveredNode(node.id);
    setTooltip({ node, x: e.clientX, y: e.clientY });
  }, []);

  const handleMouseLeave = useCallback(() => {
    setHoveredNode(null);
    setTooltip(null);
  }, []);

  if (!data) return null;

  return (
    <div ref={containerRef} style={{ width: '100%', height: '100%', position: 'relative', overflow: 'hidden' }}>
      <svg
        viewBox={`0 0 ${safeWidth} ${safeHeight}`}
        preserveAspectRatio="xMidYMid meet"
        style={{ width: '100%', height: '100%' }}
      >
        <defs>
          {/* Glow filters */}
          {['amber', 'emerald', 'rose'].map(name => {
            const colors = { amber: '#f59e0b', emerald: '#10b981', rose: '#ef4444' };
            return (
              <filter key={name} id={`glow-${name}`} x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            );
          })}
        </defs>

        {[0.2, 0.4, 0.6, 0.8].map((t) => (
          <line
            key={t}
            x1="18"
            y1={safeHeight * t}
            x2={safeWidth - 18}
            y2={safeHeight * t}
            stroke="rgba(255,255,255,0.04)"
            strokeWidth="1"
          />
        ))}

        {/* Edges */}
        {edges.map((edge, i) => {
          const sp = positions[edge.source];
          const tp = positions[edge.target];
          if (!sp || !tp) return null;
          const srcNode = nodes.find(n => n.id === edge.source);
          const tgtNode = nodes.find(n => n.id === edge.target);
          const isDebunkedPath = srcNode?.type === 'debunked' || tgtNode?.type === 'debunked';
          const isHighlighted = hoveredNode === edge.source || hoveredNode === edge.target;

          return (
            <line
              key={i}
              x1={sp.x} y1={sp.y} x2={tp.x} y2={tp.y}
              stroke={isHighlighted ? 'rgba(255,255,255,0.5)' : isDebunkedPath ? 'rgba(16,185,129,0.3)' : 'rgba(255,255,255,0.12)'}
              strokeWidth={isHighlighted ? 1.5 : isDebunkedPath ? 1 : 1}
              strokeDasharray={isDebunkedPath ? '4,3' : undefined}
            />
          );
        })}

        {/* Nodes */}
        {nodes.map(node => {
          const pos = positions[node.id];
          if (!pos) return null;
          const r = getRadius(node);
          const color = getColor(node);
          const isContained = node.type === 'infected' && node.depth === -1;
          const filterMap = { root: 'glow-amber', debunked: 'glow-emerald', infected: 'glow-rose' };
          const glowFilter = !isContained ? filterMap[node.type] : undefined;

          return (
            <g
              key={node.id}
              onMouseEnter={e => handleMouseEnter(node, e)}
              onMouseLeave={handleMouseLeave}
              style={{ cursor: 'pointer' }}
            >
              {/* Outer ring on hover */}
              {hoveredNode === node.id && (
                <circle cx={pos.x} cy={pos.y} r={r + 5}
                  fill="none" stroke={color} strokeWidth="1" opacity="0.4" />
              )}

              <circle
                cx={pos.x} cy={pos.y} r={r}
                fill={color}
                filter={glowFilter ? `url(#${glowFilter})` : undefined}
                opacity={isContained ? 0.4 : 1}
                style={{ transition: 'r 0.15s ease' }}
              />

              {/* Labels for root and debunked */}
              {node.type === 'root' && (
                <text x={pos.x} y={pos.y} fill="white" fontSize="6.5" textAnchor="middle" dominantBaseline="middle" fontWeight="bold" style={{ pointerEvents: 'none' }}>
                  SRC
                </text>
              )}
              {node.type === 'debunked' && (
                <text x={pos.x} y={pos.y - r - 5} fill="#10b981" fontSize="6" textAnchor="middle" style={{ pointerEvents: 'none' }}>
                  ✓
                </text>
              )}
            </g>
          );
        })}
      </svg>

      {/* Tooltip */}
      {tooltip && (
        <div style={{
          position: 'fixed', left: tooltip.x + 12, top: tooltip.y - 10,
          background: 'rgba(15,23,42,0.95)', border: '1px solid var(--border-color)',
          borderRadius: '8px', padding: '8px 12px', zIndex: 9999,
          pointerEvents: 'none', fontSize: '0.78rem', color: 'white',
          boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
        }}>
          <div style={{ fontWeight: 600, marginBottom: '2px' }}>Node {tooltip.node.id}</div>
          <div style={{ color: getColor(tooltip.node) }}>
            {tooltip.node.type.charAt(0).toUpperCase() + tooltip.node.type.slice(1)}
          </div>
          <div style={{ color: 'var(--text-muted)', marginTop: '2px' }}>
            Depth: {tooltip.node.depth === -1 ? 'unreachable' : tooltip.node.depth} · Degree: {tooltip.node.degree}
          </div>
        </div>
      )}
    </div>
  );
};

export default GraphViz;
