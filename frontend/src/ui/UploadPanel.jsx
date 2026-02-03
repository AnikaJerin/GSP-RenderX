import React, { useState } from "react";

export default function UploadPanel({ onFileUpload }) {
  const [file, setFile] = useState(null);

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://127.0.0.1:8000/convert", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    onFileUpload(data);
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
        accept=".stl,.obj,.ply"
        onChange={(e) => setFile(e.target.files[0])}
      />
      <button onClick={handleUpload} style={{ marginTop: "8px" }}>
        Convert to GSP
      </button>
    </div>
  );
}
