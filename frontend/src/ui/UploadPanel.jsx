import React, { useState } from "react";
import { loadGSP } from "../core/loaders/GSPLoader";

export default function UploadPanel({ onFileUpload }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [targetSplats, setTargetSplats] = useState(150000);
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
    <div className="panel panel-glass upload-panel">
      <div className="panel-title">Upload + Synthesize</div>
      <div className="panel-sub">
        Drop a mesh or archive. The engine will generate edge-aware splats in real time.
      </div>

      <label className="file-drop">
        <input
          type="file"
          accept=".obj,.stl,.ply,.glb,.gltf,.zip"
          onChange={(e) => setSelectedFile(e.target.files[0])}
        />
        <div className="file-meta">
          <div className="file-title">
            {selectedFile ? selectedFile.name : "Choose file"}
          </div>
          <div className="file-sub">STL, OBJ, GLB, PLY, or ZIP</div>
        </div>
      </label>

      <div className="control-row">
        <label>
          Target splats
          <input
            type="number"
            min="50000"
            max="2000000"
            step="50000"
            value={targetSplats}
            onChange={(e) => setTargetSplats(Number(e.target.value))}
          />
        </label>
      </div>

      <div className="control-row">
        <label>
          Edge angle (deg)
          <input
            type="number"
            min="5"
            max="80"
            step="1"
            value={edgeAngle}
            onChange={(e) => setEdgeAngle(Number(e.target.value))}
          />
        </label>
      </div>

      <button className="button-primary" onClick={handleUpload} disabled={loading}>
        {loading ? "Synthesizing..." : "Convert to GSP"}
      </button>
    </div>
  );
}
