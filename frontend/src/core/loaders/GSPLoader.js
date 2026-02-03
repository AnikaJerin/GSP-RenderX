export async function loadGSP(url) {
    const res = await fetch(url);
    const buffer = await res.arrayBuffer();
    const dv = new DataView(buffer);
  
    const count = dv.getUint32(4, true);
  
    const headerSize = 32;
    let offset = headerSize;
  
    const positions = new Float32Array(buffer, offset, count * 3);
    offset += count * 12;
  
    const normals = new Float32Array(buffer, offset, count * 3);
    offset += count * 12;
  
    const colors = new Uint8Array(buffer, offset, count * 3);
    offset += count * 3;
  
    const sizes = new Uint16Array(buffer, offset, count);
  
    return { positions, normals, colors, sizes };
  }
  