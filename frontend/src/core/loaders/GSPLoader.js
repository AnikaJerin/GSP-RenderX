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

async function fetchRange(url, start, end) {
  const res = await fetch(url, {
    headers: { Range: `bytes=${start}-${end}` },
  });
  if (!res.ok && res.status !== 206) {
    throw new Error(`Range request failed: ${res.status}`);
  }
  return res.arrayBuffer();
}

export async function loadGSPProgressive(
  url,
  { chunkPoints = 120000, onProgress } = {}
) {
  const headerBuf = await fetchRange(url, 0, 31);
  const dv = new DataView(headerBuf);

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

  const positionsStart = 32;
  const normalsStart = positionsStart + count * 3 * 4;
  const colorsStart = normalsStart + count * 3 * 4;
  const sizesStart = colorsStart + count * 3;

  const positionsBuf = new ArrayBuffer(count * 3 * 4);
  const normalsBuf = new ArrayBuffer(count * 3 * 4);
  const colorsBuf = new ArrayBuffer(count * 3);
  const sizesBuf = new ArrayBuffer(count * 2);

  const positionsU8 = new Uint8Array(positionsBuf);
  const normalsU8 = new Uint8Array(normalsBuf);
  const colorsU8 = new Uint8Array(colorsBuf);
  const sizesU8 = new Uint8Array(sizesBuf);

  let activeCount = 0;
  for (let p0 = 0; p0 < count; p0 += chunkPoints) {
    const p1 = Math.min(count, p0 + chunkPoints);

    const posStart = positionsStart + p0 * 12;
    const posEnd = positionsStart + p1 * 12 - 1;
    const norStart = normalsStart + p0 * 12;
    const norEnd = normalsStart + p1 * 12 - 1;
    const colStart = colorsStart + p0 * 3;
    const colEnd = colorsStart + p1 * 3 - 1;
    const sizStart = sizesStart + p0 * 2;
    const sizEnd = sizesStart + p1 * 2 - 1;

    const [posBuf, norBuf, colBuf, sizBuf] = await Promise.all([
      fetchRange(url, posStart, posEnd),
      fetchRange(url, norStart, norEnd),
      fetchRange(url, colStart, colEnd),
      fetchRange(url, sizStart, sizEnd),
    ]);

    positionsU8.set(new Uint8Array(posBuf), p0 * 12);
    normalsU8.set(new Uint8Array(norBuf), p0 * 12);
    colorsU8.set(new Uint8Array(colBuf), p0 * 3);
    sizesU8.set(new Uint8Array(sizBuf), p0 * 2);

    activeCount = p1;

    if (onProgress) {
      onProgress({
        positions: new Float32Array(positionsBuf),
        normals: new Float32Array(normalsBuf),
        colors: new Uint8Array(colorsBuf),
        sizes: new Uint16Array(sizesBuf),
        bboxMin,
        bboxMax,
        count,
        activeCount,
      });
    }
  }

  return {
    positions: new Float32Array(positionsBuf),
    normals: new Float32Array(normalsBuf),
    colors: new Uint8Array(colorsBuf),
    sizes: new Uint16Array(sizesBuf),
    bboxMin,
    bboxMax,
    count,
    activeCount: count,
  };
}
