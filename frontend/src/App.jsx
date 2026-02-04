import React, { useMemo, useState } from "react";
import Viewer from "./core/Viewer";
import UploadPanel from "./ui/UploadPanel";
import InfoPanel from "./ui/InfoPanel";
import TopBar from "./ui/TopBar";
import "./App.css";

export default function App() {
  const [gaussianData, setGaussianData] = useState(null);
  const [selection, setSelection] = useState(null);
  const [inspectMode, setInspectMode] = useState(false);

  const stats = useMemo(() => {
    if (!gaussianData) return null;
    return {
      total: gaussianData.count || gaussianData.positions.length / 3,
    };
  }, [gaussianData]);

  return (
    <div className="app-shell">
      <Viewer
        data={gaussianData}
        onSelect={setSelection}
        inspectMode={inspectMode}
      />

      <div className="hud">
        <TopBar stats={stats} />

        <div className="hud-columns">
          <div className="hud-left">
            <UploadPanel onFileUpload={setGaussianData} />
          </div>

          <div className="hud-spacer" />

          <div className="hud-right">
            <InfoPanel
              selection={selection}
              data={gaussianData}
              inspectMode={inspectMode}
              onToggleInspect={() => setInspectMode((v) => !v)}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
