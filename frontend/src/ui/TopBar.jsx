import React from "react";

export default function TopBar({ stats }) {
  return (
    <div className="topbar panel panel-glass">
      <div className="brand">
        <div className="brand-mark">GSP</div>
        <div>
          <div className="brand-title">GSP-RenderX</div>
          <div className="brand-sub">Edge-Preserving 3D Synthesis Engine</div>
        </div>
      </div>
      <div className="status">
        <span className="pill">V2 Research Build</span>
        {stats && (
          <span className="pill pill-accent">
            {Intl.NumberFormat().format(stats.total)} splats
          </span>
        )}
      </div>
    </div>
  );
}
