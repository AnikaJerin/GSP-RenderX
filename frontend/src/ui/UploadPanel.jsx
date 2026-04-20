import React, { useEffect, useState } from "react";
import { loadGSPProgressive } from "../core/loaders/GSPLoader";

const API_ROOT = "http://127.0.0.1:8000";

export default function UploadPanel({ onFileUpload, onVesselAnalyze }) {
  const [textPrompt, setTextPrompt] = useState("");
  const [promptLoading, setPromptLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [targetSplats, setTargetSplats] = useState(150000);
  const [edgeAngle, setEdgeAngle] = useState(35);
  const [loading, setLoading] = useState(false);

  const [imageFile, setImageFile] = useState(null);
  const [imageLoadingFace, setImageLoadingFace] = useState(false);
  const [imageLoadingMedical, setImageLoadingMedical] = useState(false);

  const [vesselFile, setVesselFile] = useState(null);
  const [vesselThreshold, setVesselThreshold] = useState(0.5);
  const [vesselMaxPoints, setVesselMaxPoints] = useState(120000);
  const [vesselLoading, setVesselLoading] = useState(false);
  const [vesselAnalyzing, setVesselAnalyzing] = useState(false);
  const [vesselContract, setVesselContract] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function loadContract() {
      try {
        const res = await fetch(`${API_ROOT}/vessel_model_contract`);
        if (!res.ok) return;
        const payload = await res.json();
        if (!cancelled) {
          setVesselContract(payload);
        }
      } catch (err) {
        console.warn("Could not load vessel contract", err);
      }
    }

    loadContract();
    return () => {
      cancelled = true;
    };
  }, []);

  const streamGspResult = async (result) => {
    if (!result.gsp_url) {
      throw new Error("Backend did not return gsp_url");
    }

    const gsp = await loadGSPProgressive(`${API_ROOT}${result.gsp_url}`, {
      chunkPoints: 120000,
      onProgress: (partial) => {
        if (onFileUpload) {
          onFileUpload({
            gsp: partial,
            meshUrl: result.mesh_url ? `${API_ROOT}${result.mesh_url}` : null,
            meshType: result.mesh_type || null,
            metadata: result.metadata || null,
            analysis:
              result.structural_features != null
                ? {
                    structural_features: result.structural_features,
                  }
                : null,
          });
        }
      },
    });

    if (onFileUpload) {
      onFileUpload({
        gsp,
        meshUrl: result.mesh_url ? `${API_ROOT}${result.mesh_url}` : null,
        meshType: result.mesh_type || null,
        metadata: result.metadata || null,
        analysis:
          result.structural_features != null
            ? {
                structural_features: result.structural_features,
              }
            : null,
      });
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert("Please select a mesh file first");
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
      const res = await fetch(`${API_ROOT}/convert?${params.toString()}`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(errText || "Upload failed");
      }

      const result = await res.json();
      await streamGspResult(result);
    } catch (err) {
      console.error(err);
      alert(err.message || "Upload failed");
    } finally {
      setLoading(false);
    }
  };

  const handlePromptGenerate = async () => {
    if (!textPrompt.trim()) {
      alert("Please enter a text prompt first");
      return;
    }

    setPromptLoading(true);
    try {
      const res = await fetch(`${API_ROOT}/generate_from_text`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: textPrompt.trim(),
          target_splats: targetSplats,
          edge_angle: edgeAngle,
          edge_oversample: 1.5,
          generator: "heuristic_mesh",
        }),
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(errText || "Prompt generation failed");
      }

      const result = await res.json();
      await streamGspResult(result);
    } catch (err) {
      console.error(err);
      alert(err.message || "Prompt generation failed");
    } finally {
      setPromptLoading(false);
    }
  };

  const handleImageReconstruct = async (mode) => {
    if (!imageFile) {
      alert("Please select an image first");
      return;
    }

    const formData = new FormData();
    formData.append("file", imageFile);

    const setImageLoading =
      mode === "fastavatar" ? setImageLoadingFace : setImageLoadingMedical;

    setImageLoading(true);
    try {
      const params = new URLSearchParams({
        target_splats: String(Math.min(Math.max(targetSplats, 1000), 500000)),
      });
      const endpoint =
        mode === "fastavatar"
          ? "reconstruct_image_fastavatar"
          : "reconstruct_image_pixel3dmm";
      const res = await fetch(`${API_ROOT}/${endpoint}?${params.toString()}`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(errText || "Image reconstruction failed");
      }

      const result = await res.json();
      await streamGspResult(result);
    } catch (err) {
      console.error(err);
      alert(err.message || "Image reconstruction failed");
    } finally {
      setImageLoading(false);
    }
  };

  const handleVesselConvert = async () => {
    if (!vesselFile) {
      alert("Please select a Colab-exported vessel volume first");
      return;
    }

    const formData = new FormData();
    formData.append("file", vesselFile);

    setVesselLoading(true);
    try {
      const params = new URLSearchParams({
        threshold: String(vesselThreshold),
        max_points: String(vesselMaxPoints),
      });
      const res = await fetch(`${API_ROOT}/convert_vessel_volume?${params.toString()}`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(errText || "Vessel conversion failed");
      }

      const result = await res.json();
      await streamGspResult(result);
    } catch (err) {
      console.error(err);
      alert(err.message || "Vessel conversion failed");
    } finally {
      setVesselLoading(false);
    }
  };

  const handleVesselAnalyze = async () => {
    if (!vesselFile) {
      alert("Please select a Colab-exported vessel volume first");
      return;
    }

    const formData = new FormData();
    formData.append("file", vesselFile);

    setVesselAnalyzing(true);
    try {
      const params = new URLSearchParams({
        threshold: String(vesselThreshold),
      });
      const res = await fetch(`${API_ROOT}/analyze_vessel_volume?${params.toString()}`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(errText || "Vessel analysis failed");
      }

      const result = await res.json();
      if (onVesselAnalyze) {
        onVesselAnalyze(result);
      }
    } catch (err) {
      console.error(err);
      alert(err.message || "Vessel analysis failed");
    } finally {
      setVesselAnalyzing(false);
    }
  };

  return (
    <div className="panel panel-glass upload-panel">
      <div className="panel-title">Upload + Research</div>
      <div className="panel-sub">
        Mesh conversion stays local here. For Option A, train the vessel model in Colab,
        export `.npy/.npz`, and inspect the centerline-aware output in this app.
      </div>

      <div className="panel-section-title">Text to 3D</div>
      <div className="panel-sub">
        Describe a simple object and generate a local 3D proxy, Gaussian splats, and a quick mesh reconstructed from those splats.
      </div>
      <div className="control-row">
        <label>
          Text prompt
          <textarea
            rows="3"
            value={textPrompt}
            onChange={(e) => setTextPrompt(e.target.value)}
            placeholder="Examples: tall rocket, wooden chair, wide table, snowman"
          />
        </label>
      </div>
      <button
        className="button-primary"
        onClick={handlePromptGenerate}
        disabled={promptLoading}
      >
        {promptLoading ? "Generating 3D Object..." : "Generate 3D from Text"}
      </button>

      <hr className="panel-separator" />

      <div className="panel-section-title">Mesh to GSP</div>
      <label className="file-drop">
        <input
          type="file"
          accept=".obj,.stl,.ply,.glb,.gltf,.zip"
          onChange={(e) => setSelectedFile(e.target.files[0])}
        />
        <div className="file-meta">
          <div className="file-title">
            {selectedFile ? selectedFile.name : "Choose mesh file"}
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
        {loading ? "Synthesizing..." : "Convert Mesh to GSP"}
      </button>

      <hr className="panel-separator" />

      <div className="panel-section-title">Vessel Centerline Workflow</div>
      <div className="panel-sub">
        Upload Colab output to analyze topology and render a centerline-conditioned
        Gaussian representation locally.
      </div>

      <label className="file-drop">
        <input
          type="file"
          accept=".npy,.npz"
          onChange={(e) => setVesselFile(e.target.files[0])}
        />
        <div className="file-meta">
          <div className="file-title">
            {vesselFile ? vesselFile.name : "Choose vessel volume"}
          </div>
          <div className="file-sub">Preferred: `.npz` with vessel_mask or vessel_prob</div>
        </div>
      </label>

      <div className="control-row">
        <label>
          Vessel threshold
          <input
            type="number"
            min="0"
            max="1"
            step="0.05"
            value={vesselThreshold}
            onChange={(e) => setVesselThreshold(Number(e.target.value))}
          />
        </label>
      </div>

      <div className="control-row">
        <label>
          Max centerline splats
          <input
            type="number"
            min="1000"
            max="500000"
            step="1000"
            value={vesselMaxPoints}
            onChange={(e) => setVesselMaxPoints(Number(e.target.value))}
          />
        </label>
      </div>

      <button
        className="button-primary"
        onClick={handleVesselConvert}
        disabled={vesselLoading}
      >
        {vesselLoading ? "Building Vessel GSP..." : "Convert Vessel to GSP"}
      </button>
      <button
        className="button-secondary"
        onClick={handleVesselAnalyze}
        disabled={vesselAnalyzing}
      >
        {vesselAnalyzing ? "Analyzing Vessel..." : "Analyze Vessel Topology"}
      </button>

      {vesselContract && (
        <div className="panel-contract">
          <div className="panel-contract-title">Colab Contract</div>
          <div className="panel-contract-line">
            Model: {vesselContract.training_contract?.model_name || "--"}
          </div>
          <div className="panel-contract-line">
            Required keys:{" "}
            {(
              vesselContract.training_contract?.accepted_upload_formats?.[".npz"]
                ?.required_any_of || []
            ).join(", ")}
          </div>
          <div className="panel-contract-line">
            Optional keys:{" "}
            {(
              vesselContract.training_contract?.accepted_upload_formats?.[".npz"]
                ?.preferred_keys || []
            ).join(", ")}
          </div>
          <div className="panel-contract-line">
            Local model state:{" "}
            {vesselContract.learned_vessel_model_state?.status || "unavailable"}
          </div>
        </div>
      )}

      <hr className="panel-separator" />

      <div className="panel-section-title">Single Image Reconstruction</div>
      <div className="panel-sub">Optional remote reconstruction for face and medical demos.</div>
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
