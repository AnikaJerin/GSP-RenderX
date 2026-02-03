export function generateGaussianSphere(count = 5000) {
    const positions = [];
    const colors = [];
    const sizes = [];
  
    for (let i = 0; i < count; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const r = 1.2 + Math.random() * 0.1;
  
      const x = r * Math.sin(phi) * Math.cos(theta);
      const y = r * Math.sin(phi) * Math.sin(theta);
      const z = r * Math.cos(phi);
  
      positions.push(x, y, z);
      colors.push(0.2 + x * 0.5, 0.4 + y * 0.5, 1.0);
      sizes.push(20.0);
    }
  
    return { positions, colors, sizes };
  }
  