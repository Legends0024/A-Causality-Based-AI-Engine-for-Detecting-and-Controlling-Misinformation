import React, { useRef, useEffect, useState } from 'react';

const GraphViz = ({ data }) => {
  const containerRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 500, height: 500 });

  useEffect(() => {
    if (containerRef.current) {
      const { clientWidth, clientHeight } = containerRef.current;
      setDimensions({ width: clientWidth, height: clientHeight });
    }
  }, []);

  if (!data) return null;
  const { nodes, edges } = data;

  const width = dimensions.width;
  const height = dimensions.height;
  const cx = width / 2;
  const cy = height / 2;

  // Find max depth to decide radius steps
  const maxDepth = Math.max(...nodes.filter(n => n.depth !== -1).map(n => n.depth));
  // If a node has depth -1 (unreachable), place it at the outer edge
  const actualMaxDepth = maxDepth > 0 ? maxDepth : 1;
  const radiusStep = Math.min(cx, cy) * 0.85 / (actualMaxDepth + 1);

  // Group nodes by depth
  const depths = {};
  nodes.forEach(node => {
    let d = node.depth === -1 ? actualMaxDepth + 1 : node.depth;
    if (!depths[d]) depths[d] = [];
    depths[d].push(node);
  });

  const positions = {};
  
  // Calculate polar coordinates
  Object.keys(depths).forEach(d => {
    const depthNodes = depths[d];
    const r = parseInt(d) * radiusStep;
    
    // If it's the root node
    if (r === 0) {
      depthNodes.forEach(node => {
        positions[node.id] = { x: cx, y: cy };
      });
    } else {
      // Distribute nodes evenly along the circumference
      const angleStep = (2 * Math.PI) / depthNodes.length;
      depthNodes.forEach((node, i) => {
        // Offset angle slightly by depth to make it look organic
        const angle = i * angleStep + (parseInt(d) * 0.5);
        positions[node.id] = {
          x: cx + r * Math.cos(angle),
          y: cy + r * Math.sin(angle)
        };
      });
    }
  });

  const getColor = (type, isContained) => {
    if (type === 'root') return 'var(--accent-amber)';
    if (type === 'debunked') return 'var(--accent-emerald)';
    if (isContained) return '#1e293b'; // dark grayish blue for contained
    return 'var(--accent-rose)';
  };

  const getRadius = (type) => {
    if (type === 'root') return 14;
    if (type === 'debunked') return 10;
    return 6;
  };

  return (
    <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }} ref={containerRef}>
      <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
        <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: '100%' }}>
          
          {/* Edges */}
          {edges.map((edge, i) => {
            const sourcePos = positions[edge.source];
            const targetPos = positions[edge.target];
            if (!sourcePos || !targetPos) return null;
            
            // Check if this is a debunked path
            const sourceNode = nodes.find(n => n.id === edge.source);
            const isContainedPath = sourceNode?.type === 'debunked';
            
            return (
              <line 
                key={i} 
                x1={sourcePos.x} 
                y1={sourcePos.y} 
                x2={targetPos.x} 
                y2={targetPos.y} 
                stroke="rgba(255, 255, 255, 0.15)" 
                strokeWidth={isContainedPath ? "0.5" : "1"} 
              />
            );
          })}

          {/* Nodes */}
          {nodes.map(node => {
            const pos = positions[node.id];
            if (!pos) return null;
            
            // A node is considered visually contained if it wasn't the root and has NO incoming edges from an infected node?
            // Actually, for simplicity and aesthetic matching, if its type is not debunked but it is unreachable from root directly (depth=-1)
            // or we identify it as contained.
            const isContained = node.type !== 'root' && node.type !== 'debunked' && node.depth === -1;
            
            const r = getRadius(node.type);
            const fill = getColor(node.type, isContained);
            
            return (
              <g key={node.id}>
                <circle 
                  cx={pos.x} 
                  cy={pos.y} 
                  r={r} 
                  fill={fill} 
                />
                {node.type === 'debunked' && (
                  <text x={pos.x} y={pos.y - r - 4} fill="white" fontSize="8" textAnchor="middle" opacity="0.8">
                    DEBUNKED
                  </text>
                )}
                {node.type === 'root' && (
                  <text x={pos.x} y={pos.y} fill="white" fontSize="7" textAnchor="middle" dominantBaseline="middle" fontWeight="bold">
                    SOURCE
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
};

export default GraphViz;