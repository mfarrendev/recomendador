import { Azar } from "../azar.ts";
import { cholesky, gram, punto, resolverCholesky } from "../algebra.ts";
import { fila, traspuesta, type Dispersa } from "../datos.ts";
import type { Aporte, Explicable, Modelo } from "./modelo.ts";

export interface OpcionesAls {
  factores: number;
  /** Regularización λ sobre las normas de X e Y. */
  lambda: number;
  /** Escala de la confianza: c = 1 + α·r. */
  alfa: number;
  iteraciones: number;
  semilla: number;
  /** Si se da, se llama con la pérdida tras cada medio paso (usuarios, ítems). */
  alPaso?: (perdida: number, paso: "usuarios" | "items", iteracion: number) => void;
}

/**
 * ALS para retroalimentación implícita (Hu, Koren y Volinsky 2008).
 *
 * Cada celda (u, i) de la matriz tiene una preferencia p = 1 si hubo
 * interacción y 0 si no, con confianza c = 1 + α·r. Se minimiza
 *
 *   Σ_{u,i} c_ui (p_ui − x_uᵀ y_i)² + λ (Σ‖x_u‖² + Σ‖y_i‖²)
 *
 * sobre TODAS las celdas, también las vacías: el silencio es un "no" débil.
 * Fijando Y, cada x_u sale de un sistema lineal f x f:
 *
 *   (YᵀY + Yᵀ(C_u − I)Y + λI) x_u = Yᵀ C_u p_u
 *
 * El truco del artículo es que YᵀY se calcula una vez por medio paso y la
 * corrección solo recorre los ítems que u tocó, así el costo es lineal en
 * las interacciones y no en usuarios x ítems. El sistema se resuelve con
 * Cholesky propio (src/algebra.ts).
 *
 * Cada medio paso minimiza exactamente la pérdida en un bloque de variables
 * con el otro fijo, así que la pérdida no puede subir; la prueba lo verifica.
 */
export class Als implements Modelo, Explicable {
  readonly nombre = "ALS implícito";

  readonly f: number;
  readonly x: Float64Array;
  readonly y: Float64Array;
  readonly nItems: number;
  readonly historia: Dispersa;
  readonly lambda: number;
  readonly alfa: number;

  constructor(f: number, x: Float64Array, y: Float64Array, nItems: number, historia: Dispersa, lambda: number, alfa: number) {
    this.f = f;
    this.x = x;
    this.y = y;
    this.nItems = nItems;
    this.historia = historia;
    this.lambda = lambda;
    this.alfa = alfa;
  }

  static entrenar(r: Dispersa, opciones: Partial<OpcionesAls> = {}): Als {
    const o = { factores: 32, lambda: 0.1, alfa: 20, iteraciones: 15, semilla: 1, ...opciones };
    const f = o.factores;
    const azar = new Azar(o.semilla);
    const x = new Float64Array(r.filas * f);
    const y = new Float64Array(r.columnas * f);
    for (let k = 0; k < y.length; k++) y[k] = 0.01 * azar.normal();
    for (let k = 0; k < x.length; k++) x[k] = 0.01 * azar.normal();
    const rt = traspuesta(r);
    for (let it = 0; it < o.iteraciones; it++) {
      pasoAls(r, y, x, f, o.lambda, o.alfa);
      o.alPaso?.(perdidaAls(r, x, y, f, o.lambda, o.alfa), "usuarios", it);
      pasoAls(rt, x, y, f, o.lambda, o.alfa);
      o.alPaso?.(perdidaAls(r, x, y, f, o.lambda, o.alfa), "items", it);
    }
    // Un último paso de usuarios: deja cada x_u como la solución exacta con el
    // Y final, que es lo que la explicación supone.
    pasoAls(r, y, x, f, o.lambda, o.alfa);
    return new Als(f, x, y, r.columnas, r, o.lambda, o.alfa);
  }

  puntuar(usuario: number, salida: Float64Array): void {
    const f = this.f;
    for (let i = 0; i < this.nItems; i++) salida[i] = punto(this.x, usuario * f, this.y, i * f, f);
  }

  /**
   * Explicación exacta de Hu et al. (sección 5): con W_u = (YᵀC_uY + λI)⁻¹,
   *   puntaje(u, i) = Σ_j c_uj · y_iᵀ W_u y_j
   * sobre los j que u tocó. Cada término es el aporte de j, y la suma de
   * todos los aportes es exactamente el puntaje (la prueba lo comprueba).
   */
  explicar(usuario: number, item: number, cuantos = 3): Aporte[] {
    return this.aportes(usuario, item).sort((a, b) => b.aporte - a.aporte).slice(0, cuantos);
  }

  aportes(usuario: number, item: number): Aporte[] {
    const f = this.f;
    const h = fila(this.historia, usuario);
    const a = sistemaFila(gram(this.y, this.nItems, f), this.y, f, h.indices, h.valores, this.lambda, this.alfa);
    const w = resolverCholesky(cholesky(a, f), f, this.y.subarray(item * f, item * f + f));
    const out: Aporte[] = [];
    for (let k = 0; k < h.indices.length; k++) {
      const j = h.indices[k]!;
      out.push({ item: j, aporte: (1 + this.alfa * h.valores[k]!) * punto(w, 0, this.y, j * f, f) });
    }
    return out;
  }
}

/** A = G + Σ_obs (c−1) y yᵀ + λI para una fila. */
function sistemaFila(
  g: Float64Array,
  y: Float64Array,
  f: number,
  indices: Int32Array,
  valores: Float64Array,
  lambda: number,
  alfa: number,
): Float64Array {
  const a = Float64Array.from(g);
  for (let d = 0; d < f; d++) a[d * f + d]! += lambda;
  for (let k = 0; k < indices.length; k++) {
    const o = indices[k]! * f;
    const cm1 = alfa * valores[k]!;
    for (let i = 0; i < f; i++) {
      const v = cm1 * y[o + i]!;
      for (let j = 0; j < f; j++) a[i * f + j]! += v * y[o + j]!;
    }
  }
  return a;
}

/**
 * Un medio paso: recalcula cada fila de `salida` (usuarios o ítems) con `fijo`
 * constante. `r` tiene por filas las entidades que se actualizan.
 */
function pasoAls(r: Dispersa, fijo: Float64Array, salida: Float64Array, f: number, lambda: number, alfa: number): void {
  const g = gram(fijo, r.columnas, f);
  const b = new Float64Array(f);
  for (let u = 0; u < r.filas; u++) {
    const h = fila(r, u);
    const a = sistemaFila(g, fijo, f, h.indices, h.valores, lambda, alfa);
    b.fill(0);
    for (let k = 0; k < h.indices.length; k++) {
      const c = 1 + alfa * h.valores[k]!;
      const o = h.indices[k]! * f;
      for (let d = 0; d < f; d++) b[d]! += c * fijo[o + d]!;
    }
    salida.set(resolverCholesky(cholesky(a, f), f, b), u * f);
  }
}

/**
 * La pérdida completa, sin recorrer usuarios x ítems: con c = 1 y p = 0 en
 * todas las celdas el término es Σ_u x_uᵀ (YᵀY) x_u; después se corrigen las
 * celdas observadas, donde c y p son otros.
 */
export function perdidaAls(r: Dispersa, x: Float64Array, y: Float64Array, f: number, lambda: number, alfa: number): number {
  const g = gram(y, r.columnas, f);
  let total = 0;
  for (let u = 0; u < r.filas; u++) {
    const o = u * f;
    for (let i = 0; i < f; i++) {
      const xi = x[o + i]!;
      for (let j = 0; j < f; j++) total += xi * g[i * f + j]! * x[o + j]!;
    }
    const h = fila(r, u);
    for (let k = 0; k < h.indices.length; k++) {
      const s = punto(x, o, y, h.indices[k]! * f, f);
      const c = 1 + alfa * h.valores[k]!;
      total += c * (1 - s) * (1 - s) - s * s;
    }
  }
  let reg = 0;
  for (const v of x) reg += v * v;
  for (const v of y) reg += v * v;
  return total + lambda * reg;
}
