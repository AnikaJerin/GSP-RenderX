import React, { useEffect } from "react"; // 👈 Add 'React' here
export default function ControlPanel() {
    return (
      <div
        style={{
          position: "absolute",
          top: 20,
          left: 20,
          color: "white",
          fontFamily: "monospace",
          pointerEvents: "none" // 👈 THIS FIXES ROTATION
        }}
      >
        <h2>GSP-Render</h2>
        <p>Phase 0: Engine Initialized</p>
      </div>
    );
  }
  