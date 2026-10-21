import { Azar } from "../azar.ts";
import { fila, type Dispersa } from "../datos.ts";
import type { Modelo } from "./modelo.ts";

export interface OpcionesBpr {
  factores: number;
  tasa: number;
  regularizacion: number;
  epocas: number;
  semilla: number;
}

/**
 * BPR-MF (Rendle et al. 2009). En vez de reconstruir la matriz, aprende a
 * ORDENAR: para un usuario u, un ítem visto i y uno no visto j, maximiza
 * ln σ(x̂_ui − x̂_uj) − λ‖Θ‖², con x̂_ui = x_uᵀy_i + b_i. Se entrena con SGD
 * sobre tríos muestreados al azar (bootstrap): una época son tantos tríos
 * como interacciones hay. El j se muestrea uniforme entre los no vistos.
 */
export class Bpr implements Modelo {
  readonly nombre = "BPR";

  readonly f: number;
  readonly x: Float64Array;
  readonly y: Float64Array;
  readonly sesgo: Float64Array;

  constructor(f: number, x: Float64Array, y: Float64Array, sesgo: Float64Array) {
    this.f = f;
    this.x = x;
    this.y = y;
    this.sesgo = sesgo;
  }

  static entrenar(r: Dispersa, opciones: Partial<OpcionesBpr> = {}): Bpr {
    const o: OpcionesBpr = { factores: 32, tasa: 0.05, regularizacion: 0.002, epocas: 40, semilla: 1, ...opciones };
    const f = o.factores;
    const azar = new Azar(o.semilla);
    const x = new Float64Array(r.filas * f).map(() => 0.1 * azar.normal());
    const y = new Float64Array(r.columnas * f).map(() => 0.1 * azar.normal());
    const sesgo = new Float64Array(r.columnas);
    const vistos = Array.from({ length: r.filas }, (_, u) => new Set(fila(r, u).indices));
    // Para muestrear (u, i) uniforme entre interacciones: el usuario de cada posición.
    const usuarioDe = new Int32Array(r.indices.length);
    for (let u = 0; u < r.filas; u++) for (let p = r.inicio[u]!; p < r.inicio[u + 1]!; p++) usuarioDe[p] = u;
    const nnz = r.indices.length;
    const lr = o.tasa;
    const reg = o.regularizacion;
    for (let e = 0; e < o.epocas; e++) {
      for (let paso = 0; paso < nnz; paso++) {
        const p = azar.entero(nnz);
        const u = usuarioDe[p]!;
        const i = r.indices[p]!;
        if (vistos[u]!.size >= r.columnas) continue;
        let j = azar.entero(r.columnas);
        while (vistos[u]!.has(j)) j = azar.entero(r.columnas);
        const ou = u * f;
        const oi = i * f;
        const oj = j * f;
        let diff = sesgo[i]! - sesgo[j]!;
        for (let d = 0; d < f; d++) diff += x[ou + d]! * (y[oi + d]! - y[oj + d]!);
        // d/dθ ln σ(diff) = σ(−diff) · d diff/dθ
        const g = 1 / (1 + Math.exp(diff));
        for (let d = 0; d < f; d++) {
          const xu = x[ou + d]!;
          const yi = y[oi + d]!;
          const yj = y[oj + d]!;
          x[ou + d] = xu + lr * (g * (yi - yj) - reg * xu);
          y[oi + d] = yi + lr * (g * xu - reg * yi);
          y[oj + d] = yj + lr * (-g * xu - reg * yj);
        }
        sesgo[i] = sesgo[i]! + lr * (g - reg * sesgo[i]!);
        sesgo[j] = sesgo[j]! + lr * (-g - reg * sesgo[j]!);
      }
    }
    return new Bpr(f, x, y, sesgo);
  }

  puntuar(usuario: number, salida: Float64Array): void {
    const f = this.f;
    const o = usuario * f;
    for (let i = 0; i < this.sesgo.length; i++) {
      let s = this.sesgo[i]!;
      for (let d = 0; d < f; d++) s += this.x[o + d]! * this.y[i * f + d]!;
      salida[i] = s;
    }
  }
}
