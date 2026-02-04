export async function loadGSP(url) {
  const res = await fetch(url);
  const buffer = await res.arrayBuffer();
  const dv = new DataView(buffer);

  const magic =
    String.fromCharCode(dv.getUint8(0)) +
    String.fromCharCode(dv.getUint8(1)) +
    String.fromCharCode(dv.getUint8(2)) +
    String.fromCharCode(dv.getUint8(3));
  if (magic !== "GSP1") {
    throw new Error("Invalid GSP file");
  }

  const count = dv.getUint32(4, true);
  const bboxMin = [
    dv.getFloat32(8, true),
    dv.getFloat32(12, true),
    dv.getFloat32(16, true),
  ];
  const bboxMax = [
    dv.getFloat32(20, true),
    dv.getFloat32(24, true),
    dv.getFloat32(28, true),
  ];
  let offset = 32;

  const positions = new Float32Array(count * 3);
  for (let i = 0; i < positions.length; i++, offset += 4) {
    positions[i] = dv.getFloat32(offset, true);
  }

  const normals = new Float32Array(count * 3);
  for (let i = 0; i < normals.length; i++, offset += 4) {
    normals[i] = dv.getFloat32(offset, true);
  }

  const colors = new Uint8Array(count * 3);
  for (let i = 0; i < colors.length; i++, offset += 1) {
    colors[i] = dv.getUint8(offset);
  }

  const sizes = new Uint16Array(count);
  for (let i = 0; i < sizes.length; i++, offset += 2) {
    sizes[i] = dv.getUint16(offset, true);
  }

  return { positions, normals, colors, sizes, bboxMin, bboxMax, count };
}
