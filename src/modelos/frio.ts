import { resolverSPD } from "../algebra.ts";
import { fila, type Dispersa } from "../datos.ts";
import type { Als } from "./als.ts";
import type { Modelo } from "./modelo.ts";

function normalizar(v: Float64Array): Float64Array {
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n);
  return n === 0 ? v.slice() : v.map((x) => x / n);
}

/**
 * Recomendación por contenido: el perfil del usuario es el promedio de los
 * atributos (normalizados) de lo que tocó, y el puntaje es el coseno entre
 * ese perfil y los atributos de cada ítem. No necesita que el ítem tenga
 * historia: sirve para un ítem recién llegado.
 */
export class Contenido implements Modelo {
  readonly nombre = "contenido";
  private readonly attr: Float64Array[];

  readonly historia: Dispersa;

  constructor(atributos: Float64Array[], historia: Dispersa) {
    this.historia = historia;
    this.attr = atributos.map(normalizar);
  }

  puntuar(usuario: number, salida: Float64Array): void {
    const d = this.attr[0]?.length ?? 0;
    const perfil = new Float64Array(d);
    const h = fila(this.historia, usuario);
    for (let k = 0; k < h.indices.length; k++) {
      const a = this.attr[h.indices[k]!]!;
      for (let t = 0; t < d; t++) perfil[t]! += a[t]!;
    }
    const p = normalizar(perfil);
    for (let i = 0; i < this.attr.length; i++) {
      let s = 0;
      const a = this.attr[i]!;
      for (let t = 0; t < d; t++) s += p[t]! * a[t]!;
      salida[i] = s;
    }
  }
}

/**
 * Híbrido para arranque en frío. ALS no tiene factor para un ítem sin
 * interacciones (su y_i quedó en lo que la regularización deja: casi cero).
 * Se aprende una regresión ridge de atributos → factores sobre los ítems con
 * historia, Y_tibio ≈ [A 1] W, y con ella se le da a cada ítem frío un
 * factor y_i = [a_i 1] W. Los ítems tibios conservan el suyo, así el
 * híbrido es ALS donde hay datos y contenido proyectado donde no.
 */
export class Hibrido implements Modelo {
  readonly nombre = "híbrido (ALS + atributos)";
  readonly y: Float64Array;

  readonly als: Als;

  constructor(als: Als, atributos: Float64Array[], frios: ReadonlySet<number>, lambda = 1) {
    this.als = als;
    const f = als.f;
    const d = (atributos[0]?.length ?? 0) + 1;
    const fila1 = (i: number) => {
      const v = new Float64Array(d);
      v.set(atributos[i]!);
      v[d - 1] = 1;
      return v;
    };
    // Ecuaciones normales: (AᵀA + λI) W = AᵀY, resueltas columna a columna.
    const ata = new Float64Array(d * d);
    const aty = new Float64Array(d * f);
    for (let i = 0; i < als.nItems; i++) {
      if (frios.has(i)) continue;
      const a = fila1(i);
      for (let p = 0; p < d; p++) {
        for (let q = 0; q < d; q++) ata[p * d + q]! += a[p]! * a[q]!;
        for (let c = 0; c < f; c++) aty[p * f + c]! += a[p]! * als.y[i * f + c]!;
      }
    }
    for (let p = 0; p < d; p++) ata[p * d + p]! += lambda;
    const w = new Float64Array(d * f);
    for (let c = 0; c < f; c++) {
      const b = new Float64Array(d);
      for (let p = 0; p < d; p++) b[p] = aty[p * f + c]!;
      const sol = resolverSPD(ata, d, b);
      for (let p = 0; p < d; p++) w[p * f + c] = sol[p]!;
    }
    this.y = Float64Array.from(als.y);
    for (const i of frios) {
      const a = fila1(i);
      for (let c = 0; c < f; c++) {
        let s = 0;
        for (let p = 0; p < d; p++) s += a[p]! * w[p * f + c]!;
        this.y[i * f + c] = s;
      }
    }
  }

  puntuar(usuario: number, salida: Float64Array): void {
    const f = this.als.f;
    const o = usuario * f;
    for (let i = 0; i < this.als.nItems; i++) {
      let s = 0;
      for (let c = 0; c < f; c++) s += this.als.x[o + c]! * this.y[i * f + c]!;
      salida[i] = s;
    }
  }
}
