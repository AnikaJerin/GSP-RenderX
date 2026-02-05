import React, { useState } from "react";
import { loadGSPProgressive } from "../core/loaders/GSPLoader";

export default function UploadPanel({ onFileUpload }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [targetSplats, setTargetSplats] = useState(150000);
  const [edgeAngle, setEdgeAngle] = useState(35);
  const [loading, setLoading] = useState(false);
  const [imageFile, setImageFile] = useState(null);
  const [imageLoadingFace, setImageLoadingFace] = useState(false);
  const [imageLoadingMedical, setImageLoadingMedical] = useState(false);
  const handleUpload = async () => {
    if (!selectedFile) {
      alert("Please select a file first");
      return;
    }
  
    const formData = new FormData();
    formData.append("file", selectedFile); 

    setLoading(true);
    try {
      const params = new URLSearchParams({
        target_splats: String(targetSplats),
        edge_angle: String(edgeAngle),
      });
      const res = await fetch(`http://127.0.0.1:8000/convert?${params.toString()}`, {
        method: "POST",
        body: formData, 
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

      const gsp = await loadGSPProgressive(`http://127.0.0.1:8000${result.gsp_url}`, {
        chunkPoints: 120000,
        onProgress: (partial) => {
          if (onFileUpload) {
            onFileUpload({
              gsp: partial,
              meshUrl: result.mesh_url ? `http://127.0.0.1:8000${result.mesh_url}` : null,
              meshType: result.mesh_type || null,
            });
          }
        },
      });
      if (onFileUpload) {
        onFileUpload({
          gsp,
          meshUrl: result.mesh_url ? `http://127.0.0.1:8000${result.mesh_url}` : null,
          meshType: result.mesh_type || null,
        });
      }
    } catch (err) {
      console.error(err);
      alert(err.message || "Upload failed");
    } finally {
      setLoading(false);
    }
  };

  const handleImageReconstruct = async (mode) => {
    if (!imageFile) {
      alert("Please select an image first");
      return;
    }

    const formData = new FormData();
    formData.append("file", imageFile);

    const setLoading =
      mode === "fastavatar" ? setImageLoadingFace : setImageLoadingMedical;

    setLoading(true);
    try {
      const params = new URLSearchParams({
        target_splats: String(Math.min(Math.max(targetSplats, 1000), 500000)),
      });
      const endpoint =
        mode === "fastavatar"
          ? "reconstruct_image_fastavatar"
          : "reconstruct_image_pixel3dmm";
      const res = await fetch(
        `http://127.0.0.1:8000/${endpoint}?${params.toString()}`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(errText || "Image reconstruction failed");
      }
      const result = await res.json();
      console.log("Image reconstruction returned:", result);

      if (!result.gsp_url) {
        console.error("Backend did not return gsp_url");
        return;
      }

      const gsp = await loadGSPProgressive(
        `http://127.0.0.1:8000${result.gsp_url}`,
        {
          chunkPoints: 120000,
          onProgress: (partial) => {
            if (onFileUpload) {
              onFileUpload({
                gsp: partial,
                meshUrl: null,
                meshType: null,
              });
            }
          },
        }
      );
      if (onFileUpload) {
        onFileUpload({
          gsp,
          meshUrl: null,
          meshType: null,
        });
      }
    } catch (err) {
      console.error(err);
      alert(err.message || "Image reconstruction failed");
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

      <hr className="panel-separator" />
      <div className="panel-sub">
        Or reconstruct from a single reference image.
      </div>
      <div className="control-row">
        <label>
          Reference image
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setImageFile(e.target.files[0])}
          />
        </label>
      </div>
      <button
        className="button-secondary"
        onClick={() => handleImageReconstruct("fastavatar")}
        disabled={imageLoadingFace}
      >
        {imageLoadingFace ? "Reconstructing (Face)..." : "Reconstruct Face (FastAvatar)"}
      </button>
      <button
        className="button-secondary"
        onClick={() => handleImageReconstruct("pixel3dmm")}
        disabled={imageLoadingMedical}
      >
        {imageLoadingMedical
          ? "Reconstructing (Medical)..."
          : "Reconstruct Medical (Pixel3DMM)"}
      </button>
    </div>
  );
}
