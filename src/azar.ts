/**
 * Generador pseudoaleatorio con semilla (mulberry32). Todo lo que usa azar en
 * este proyecto (datos sintéticos, inicialización de factores, muestreo de
 * BPR) recibe una instancia de esto: la misma semilla da el mismo resultado,
 * y eso es lo que permite que la tabla del README se pueda reproducir.
 */
export class Azar {
  private estado: number;

  constructor(semilla: number) {
    this.estado = semilla >>> 0;
  }

  /** Uniforme en [0, 1). */
  siguiente(): number {
    this.estado = (this.estado + 0x6d2b79f5) >>> 0;
    let t = this.estado;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Entero uniforme en [0, n). */
  entero(n: number): number {
    return Math.floor(this.siguiente() * n);
  }

  /** Normal estándar (Box-Muller). */
  normal(): number {
    let u = 0;
    while (u === 0) u = this.siguiente();
    const v = this.siguiente();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /** Índice muestreado según pesos acumulados (búsqueda binaria). */
  ponderado(acumulados: Float64Array): number {
    const total = acumulados[acumulados.length - 1]!;
    const x = this.siguiente() * total;
    let lo = 0;
    let hi = acumulados.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (acumulados[mid]! > x) hi = mid;
      else lo = mid + 1;
    }
    return lo;
  }
}
