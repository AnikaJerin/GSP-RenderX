import React from "react";

export default function InfoPanel({
  selection,
  data,
  inspectMode,
  onToggleInspect,
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

      <div className="meta-grid">
        <div className="meta-item">
          <div className="meta-label">Selection</div>
          <div className="meta-value">
            {selection ? `#${selection.index}` : "None"}
          </div>
        </div>
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
