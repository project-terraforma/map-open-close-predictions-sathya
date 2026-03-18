import MapContainer from './components/MapContainer';

function App() {
  return (
    <div className="h-screen w-screen bg-slate-950 overflow-hidden relative">
      <MapContainer />

      {/* Subtle Overlay for Branding */}
      <div className="absolute top-4 left-4 z-[1000] bg-slate-900/80 backdrop-blur-md px-4 py-2 rounded-full border border-white/10 shadow-xl pointer-events-none">
        <h1 className="text-sm font-bold text-white flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
          TERRAFORMA: LIVE MAP
        </h1>
      </div>
    </div>
  );
}

export default App;
