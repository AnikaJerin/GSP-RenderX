import React from "react";

export default function InfoPanel({ selection, data }) {
  const count = data?.count || (data?.positions ? data.positions.length / 3 : 0);

  return (
    <div className="panel panel-glass info-panel">
      <div className="panel-title">Inspector</div>
      <div className="panel-sub">
        {count
          ? `${Intl.NumberFormat().format(count)} active splats`
          : "Upload a model to begin."}
      </div>

      <div className="meta-grid">
        <div className="meta-item">
          <div className="meta-label">Selection</div>
          <div className="meta-value">
            {selection ? `#${selection.index}` : "None"}
          </div>
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
      </div>

      <div className="panel-callout">
        Next: attach metadata labels to edge clusters and parts.
      </div>
    </div>
  );
}
