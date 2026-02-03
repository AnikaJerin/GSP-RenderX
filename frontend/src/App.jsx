import React from "react";
import Viewer from "./core/Viewer";
import ControlPanel from "./ui/ControlPanel";

export default function App() {
  return (
    <div style={{ width: "100vw", height: "100vh", position: "relative" }}>
      <Viewer />
      <ControlPanel />
    </div>
  );
}
