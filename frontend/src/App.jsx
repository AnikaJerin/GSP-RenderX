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
  const [measureMode, setMeasureMode] = useState(false);
  const [measurePoints, setMeasurePoints] = useState([]);
  const [clipEnabled, setClipEnabled] = useState(false);
  const [clipAxis, setClipAxis] = useState("y");
  const [clipRatio, setClipRatio] = useState(0.5);
  const [showFps, setShowFps] = useState(false);
  const [fps, setFps] = useState(0);
  const [renderStats, setRenderStats] = useState(null);
  const [annotationMode, setAnnotationMode] = useState(false);
  const [annotations, setAnnotations] = useState([]);
  const [annotationLabel, setAnnotationLabel] = useState("Note");
  const [showAnnotations, setShowAnnotations] = useState(true);

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
        onSelect={(point) => {
          setSelection(point);
          if (measureMode) {
            setMeasurePoints((prev) => {
              const next = [...prev, point].slice(-2);
              return next;
            });
          }
          if (annotationMode) {
            setAnnotations((prev) => [
              ...prev,
              {
                id: `${Date.now()}-${prev.length + 1}`,
                label: `${annotationLabel} ${prev.length + 1}`,
                position: point.position,
              },
            ]);
          }
        }}
        inspectMode={inspectMode}
        clipEnabled={clipEnabled}
        clipAxis={clipAxis}
        clipRatio={clipRatio}
        showFps={showFps}
        onFps={setFps}
        annotations={annotations}
        showAnnotations={showAnnotations}
        onRenderStats={setRenderStats}
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
              measureMode={measureMode}
              onToggleMeasure={() => {
                setMeasureMode((v) => !v);
                setMeasurePoints([]);
              }}
              measurePoints={measurePoints}
              clipEnabled={clipEnabled}
              onToggleClip={() => setClipEnabled((v) => !v)}
              clipAxis={clipAxis}
              onChangeClipAxis={setClipAxis}
              clipRatio={clipRatio}
              onChangeClipRatio={setClipRatio}
              showFps={showFps}
              onToggleFps={() => setShowFps((v) => !v)}
              fps={fps}
              renderStats={renderStats}
              annotationMode={annotationMode}
              onToggleAnnotation={() => setAnnotationMode((v) => !v)}
              annotationLabel={annotationLabel}
              onChangeAnnotationLabel={setAnnotationLabel}
              annotations={annotations}
              onClearAnnotations={() => setAnnotations([])}
              showAnnotations={showAnnotations}
              onToggleShowAnnotations={() => setShowAnnotations((v) => !v)}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
