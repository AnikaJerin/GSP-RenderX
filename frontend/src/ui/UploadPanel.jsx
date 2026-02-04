import React, { useState } from "react";
import { loadGSP } from "../core/loaders/GSPLoader";

export default function UploadPanel({ onFileUpload }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [targetSplats, setTargetSplats] = useState(500000);
  const [edgeAngle, setEdgeAngle] = useState(35);
  const [loading, setLoading] = useState(false);
  const handleUpload = async () => {
    if (!selectedFile) {
      alert("Please select a file first");
      return;
    }
  
    const formData = new FormData();
    formData.append("file", selectedFile); // MUST be actual File

    setLoading(true);
    try {
      const params = new URLSearchParams({
        target_splats: String(targetSplats),
        edge_angle: String(edgeAngle),
      });
      const res = await fetch(`http://127.0.0.1:8000/convert?${params.toString()}`, {
        method: "POST",
        body: formData, // DO NOT set headers
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(errText || "Upload failed");
      }
      const result = await res.json();
      console.log("Server returned:", result);

      if (!result.gsp_url) {
        console.error("Backend did not return gsp_url");
        return;
      }

      const gsp = await loadGSP(`http://127.0.0.1:8000${result.gsp_url}`);
      if (onFileUpload) {
        onFileUpload(gsp);
      }
    } catch (err) {
      console.error(err);
      alert(err.message || "Upload failed");
    } finally {
      setLoading(false);
    }
  };
   
  

  return (
    <div style={{
      position: "absolute",
      top: 20,
      right: 20,
      background: "#111",
      padding: "12px",
      borderRadius: "8px",
      color: "white",
      fontFamily: "monospace"
    }}>
      <input
        type="file"
        accept=".obj,.stl,.ply,.glb,.gltf,.zip"
        onChange={(e) => setSelectedFile(e.target.files[0])}
      />

      <div style={{ marginTop: "8px" }}>
        <label style={{ display: "block", fontSize: "12px", marginBottom: "4px" }}>
          Target splats
        </label>
        <input
          type="number"
          min="50000"
          max="2000000"
          step="50000"
          value={targetSplats}
          onChange={(e) => setTargetSplats(Number(e.target.value))}
          style={{ width: "100%" }}
        />
      </div>

      <div style={{ marginTop: "8px" }}>
        <label style={{ display: "block", fontSize: "12px", marginBottom: "4px" }}>
          Edge angle (deg)
        </label>
        <input
          type="number"
          min="5"
          max="80"
          step="1"
          value={edgeAngle}
          onChange={(e) => setEdgeAngle(Number(e.target.value))}
          style={{ width: "100%" }}
        />
      </div>

      <button onClick={handleUpload} style={{ marginTop: "8px" }}>
        {loading ? "Processing..." : "Convert to GSP"}
      </button>
    </div>
  );
}
