import React from 'react';

const GraphViz = ({ data, title }) => {
  const { nodes, edges } = data;
  // Group by depth
  const depths = {};
  nodes.forEach(node => {
    if (!depths[node.depth]) depths[node.depth] = [];
    depths[node.depth].push(node);
  });
  // Assign positions
  const positions = {};
  const depthKeys = Object.keys(depths).sort((a,b) => a - b);
  const maxDepth = Math.max(...depthKeys);
  const width = 800;
  const height = 600;
  const xStep = width / (maxDepth + 1);
  depthKeys.forEach(d => {
    const depthNodes = depths[d];
    const yStep = height / (depthNodes.length + 1);
    depthNodes.forEach((node, i) => {
      positions[node.id] = {
        x: (parseInt(d) + 1) * xStep,
        y: (i + 1) * yStep
      };
    });
  });
  const getColor = (type) => {
    if (type === 'root') return '#f59e0b';
    if (type === 'debunked') return '#10b981';
    return '#ef4444';
  };
  const getRadius = (type) => {
    if (type === 'root') return 10;
    if (type === 'debunked') return 8;
    return 5;
  };
  return (
    <div>
      <h3>{title}</h3>
      <svg width={width} height={height} style={{ border: '1px solid #1e293b', background: '#020817' }}>
        {edges.map((edge, i) => {
          const sourcePos = positions[edge.source];
          const targetPos = positions[edge.target];
          if (!sourcePos || !targetPos) return null;
          return <line key={i} x1={sourcePos.x} y1={sourcePos.y} x2={targetPos.x} y2={targetPos.y} stroke="#1e293b" strokeWidth="1" />;
        })}
        {nodes.map(node => {
          const pos = positions[node.id];
          if (!pos) return null;
          return <circle key={node.id} cx={pos.x} cy={pos.y} r={getRadius(node.type)} fill={getColor(node.type)} />;
        })}
      </svg>
      <div style={{ marginTop: '10px' }}>
        <span style={{ color: '#f59e0b' }}>● Root</span>
        <span style={{ color: '#10b981', marginLeft: '10px' }}>● Debunked</span>
        <span style={{ color: '#ef4444', marginLeft: '10px' }}>● Infected</span>
      </div>
    </div>
  );
};

export default GraphViz;