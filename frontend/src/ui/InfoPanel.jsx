import React from "react";

export default function InfoPanel({
  selection,
  data,
  inspectMode,
  onToggleInspect,
  measureMode,
  onToggleMeasure,
  measurePoints = [],
  clipEnabled,
  onToggleClip,
  clipAxis,
  onChangeClipAxis,
  clipRatio,
  onChangeClipRatio,
  showFps,
  onToggleFps,
  fps,
  renderStats,
  annotationMode,
  onToggleAnnotation,
  annotationLabel,
  onChangeAnnotationLabel,
  annotations,
  onClearAnnotations,
  showAnnotations,
  onToggleShowAnnotations,
}) {
  const count = data?.count || (data?.positions ? data.positions.length / 3 : 0);
  const bboxMin = data?.bboxMin;
  const bboxMax = data?.bboxMax;
  const center = bboxMin && bboxMax
    ? [
        (bboxMin[0] + bboxMax[0]) / 2,
        (bboxMin[1] + bboxMax[1]) / 2,
        (bboxMin[2] + bboxMax[2]) / 2,
      ]
    : null;

  let partLabel = "--";
  let partDesc = "--";
  if (selection && center) {
    const [x, y, z] = selection.position;
    const lr = x >= center[0] ? "Right" : "Left";
    const ud = y >= center[1] ? "Upper" : "Lower";
    const fb = z >= center[2] ? "Front" : "Back";
    partLabel = `${ud}-${lr}-${fb}`;
    partDesc = `Spatial cluster derived from bbox octant (${ud}, ${lr}, ${fb}).`;
  }

  const distance =
    measurePoints && measurePoints.length === 2
      ? Math.sqrt(
          Math.pow(measurePoints[0].position[0] - measurePoints[1].position[0], 2) +
            Math.pow(measurePoints[0].position[1] - measurePoints[1].position[1], 2) +
            Math.pow(measurePoints[0].position[2] - measurePoints[1].position[2], 2)
        )
      : null;

  return (
    <div className="panel panel-glass info-panel">
      <div className="panel-title">Inspector</div>
      <div className="panel-sub">
        {count
          ? `${Intl.NumberFormat().format(count)} active splats`
          : "Upload a model to begin."}
      </div>

      <button className="button-secondary" onClick={onToggleInspect}>
        {inspectMode ? "Exit Inspect Mode" : "Enable Inspect Mode"}
      </button>
      <button className="button-secondary" onClick={onToggleMeasure}>
        {measureMode ? "Exit Measure Mode" : "Measure Distance"}
      </button>
      <button className="button-secondary" onClick={onToggleAnnotation}>
        {annotationMode ? "Annotation Mode: ON" : "Annotation Mode: OFF"}
      </button>
      <button className="button-secondary" onClick={onToggleShowAnnotations}>
        {showAnnotations ? "Hide Annotations" : "Show Annotations"}
      </button>
      <div className="control-row">
        <label>
          Annotation Label
          <input
            type="text"
            value={annotationLabel}
            onChange={(e) => onChangeAnnotationLabel(e.target.value)}
          />
        </label>
      </div>
      <button className="button-secondary" onClick={onClearAnnotations}>
        Clear Annotations
      </button>
      <button className="button-secondary" onClick={onToggleClip}>
        {clipEnabled ? "Disable Section" : "Enable Section"}
      </button>
      {clipEnabled && (
        <>
          <div className="control-row">
            <label>
              Section Axis
              <select
                value={clipAxis}
                onChange={(e) => onChangeClipAxis(e.target.value)}
              >
                <option value="x">X</option>
                <option value="y">Y</option>
                <option value="z">Z</option>
              </select>
            </label>
          </div>
          <div className="control-row">
            <label>
              Section Depth
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={clipRatio}
                onChange={(e) => onChangeClipRatio(Number(e.target.value))}
              />
            </label>
          </div>
        </>
      )}
      <button className="button-secondary" onClick={onToggleFps}>
        {showFps ? "Hide FPS" : "Show FPS"}
      </button>

      <div className="meta-grid">
        <div className="meta-item">
          <div className="meta-label">Selection</div>
          <div className="meta-value">
            {selection ? `#${selection.index}` : "None"}
          </div>
        </div>
        <div className="meta-item">
          <div className="meta-label">Measure</div>
          <div className="meta-value mono">
            {distance != null ? distance.toFixed(4) : "--"}
          </div>
        </div>
        <div className="meta-item">
          <div className="meta-label">FPS</div>
          <div className="meta-value mono">{showFps ? fps.toFixed(1) : "--"}</div>
        </div>
        <div className="meta-item">
          <div className="meta-label">Splat Count</div>
          <div className="meta-value mono">
            {renderStats ? renderStats.total : data?.count || "--"}
          </div>
        </div>
        <div className="meta-item">
          <div className="meta-label">Active Splats</div>
          <div className="meta-value mono">
            {renderStats ? renderStats.active : data?.activeCount || "--"}
          </div>
        </div>
        <div className="meta-item">
          <div className="meta-label">Rendered Splats</div>
          <div className="meta-value mono">
            {renderStats ? renderStats.rendered : "--"}
          </div>
        </div>
        <div className="meta-item">
          <div className="meta-label">Annotations</div>
          <div className="meta-value mono">
            {annotations && annotations.length ? annotations.length : 0}
          </div>
        </div>
        {annotations && annotations.length > 0 && (
          <div className="meta-item">
            <div className="meta-label">Annotation Labels</div>
            <div className="meta-value mono">
              {annotations.slice(-5).map((ann) => ann.label).join(", ")}
            </div>
          </div>
        )}
        <div className="meta-item">
          <div className="meta-label">Part ID</div>
          <div className="meta-value">{partLabel}</div>
        </div>
        <div className="meta-item">
          <div className="meta-label">Position</div>
          <div className="meta-value mono">
            {selection
              ? `${selection.position[0].toFixed(3)}, ${selection.position[1].toFixed(
                  3
                )}, ${selection.position[2].toFixed(3)}`
              : "--"}
          </div>
        </div>
        <div className="meta-item">
          <div className="meta-label">Radius</div>
          <div className="meta-value mono">
            {selection ? selection.size.toFixed(4) : "--"}
          </div>
        </div>
        <div className="meta-item">
          <div className="meta-label">Color</div>
          <div className="meta-value mono">
            {selection
              ? `rgb(${selection.color[0]}, ${selection.color[1]}, ${selection.color[2]})`
              : "--"}
          </div>
        </div>
        <div className="meta-item">
          <div className="meta-label">Description</div>
          <div className="meta-value">{partDesc}</div>
        </div>
      </div>

      <div className="panel-callout">
        Tip: Inspect mode enables point picking without slowing rotation.
      </div>
    </div>
  );
}
