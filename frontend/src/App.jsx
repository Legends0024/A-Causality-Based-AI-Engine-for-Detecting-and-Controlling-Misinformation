import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import Home from './pages/Home.jsx';
import Dashboard from './pages/Dashboard.jsx';

function Navbar() {
  return (
    <nav style={{
      position: 'fixed', top: 0, left: 0, right: 0, zIndex: 1000,
      background: 'rgba(7,9,14,0.85)', backdropFilter: 'blur(12px)',
      borderBottom: '1px solid var(--border-color)',
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '0 32px', height: '56px',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <span style={{ fontSize: '1.3rem' }}>🕸️</span>
        <span style={{ fontWeight: 700, fontSize: '1rem', letterSpacing: '-0.02em' }}>
          Causal<span style={{ color: 'var(--accent-emerald)' }}>AI</span>
        </span>
      </div>
      <div style={{ display: 'flex', gap: '8px' }}>
        {[
          { to: '/', label: 'Home' },
          { to: '/dashboard', label: 'Dashboard' },
        ].map(({ to, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            style={({ isActive }) => ({
              padding: '6px 16px',
              borderRadius: '999px',
              fontSize: '0.875rem',
              fontWeight: 500,
              textDecoration: 'none',
              color: isActive ? '#fff' : 'var(--text-secondary)',
              background: isActive ? 'rgba(99,102,241,0.2)' : 'transparent',
              border: isActive ? '1px solid rgba(99,102,241,0.4)' : '1px solid transparent',
              transition: 'all 0.2s ease',
            })}
          >
            {label}
          </NavLink>
        ))}
      </div>
    </nav>
  );
}

function App() {
  return (
    <Router>
      <Navbar />
      <div style={{ paddingTop: '56px', minHeight: '100vh' }}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/dashboard" element={<Dashboard />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;