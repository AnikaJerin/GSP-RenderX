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
  const positionsBytes = count * 3 * 4;
  const normalsBytes = count * 3 * 4;
  const colorsBytes = count * 3;
  const sizesBytes = count * 2;

  const positions = new Float32Array(buffer, offset, count * 3);
  offset += positionsBytes;

  const normals = new Float32Array(buffer, offset, count * 3);
  offset += normalsBytes;

  const colors = new Uint8Array(buffer, offset, count * 3);
  offset += colorsBytes;

  let sizes;
  if (offset % 2 === 0) {
    sizes = new Uint16Array(buffer, offset, count);
  } else {
    // Fallback: copy into aligned buffer
    const aligned = buffer.slice(offset, offset + sizesBytes);
    sizes = new Uint16Array(aligned, 0, count);
  }
  offset += sizesBytes;

  return { positions, normals, colors, sizes, bboxMin, bboxMax, count };
}
