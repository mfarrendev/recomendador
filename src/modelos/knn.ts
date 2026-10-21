import { fila, type Dispersa } from "../datos.ts";
import type { Aporte, Explicable, Modelo } from "./modelo.ts";

export type Similitud = "coseno" | "ajustada";

export interface OpcionesKnn {
  similitud: Similitud;
  /** Vecinos que se guardan por ítem. */
  vecinos: number;
  /** Encogimiento: sim · n / (n + encoger), con n = usuarios en común. */
  encoger: number;
}

/**
 * Item-kNN sobre interacciones implícitas.
 *
 * - coseno: similitud coseno entre las columnas de la matriz usuario x ítem,
 *   con peso log(1 + valor) para que un ítem tocado cien veces no aplaste al
 *   resto de la historia.
 * - ajustada: coseno ajustado (Sarwar et al. 2001). A cada valor se le resta
 *   el promedio del usuario sobre lo que tocó. Solo tiene sentido si los
 *   valores varían (cuentas, notas de MovieLens); con datos binarios todo se
 *   vuelve cero y la similitud desaparece, por eso no se usa en esos casos.
 *
 * Puntaje(u, i) = suma de sim(j, i) sobre los j de la historia de u, donde i
 * está entre los `vecinos` más parecidos a j. Esa misma suma, separada por j,
 * es la explicación.
 */
export class ItemKnn implements Modelo, Explicable {
  readonly nombre: string;

  readonly historia: Dispersa;
  /** Por ítem j: sus vecinos y la similitud con cada uno. */
  readonly vecinosDe: { items: Int32Array; sims: Float64Array }[];

  constructor(historia: Dispersa, vecinosDe: { items: Int32Array; sims: Float64Array }[], similitud: Similitud) {
    this.historia = historia;
    this.vecinosDe = vecinosDe;
    this.nombre = `item-kNN (${similitud})`;
  }

  static entrenar(r: Dispersa, opciones: Partial<OpcionesKnn> = {}): ItemKnn {
    const o: OpcionesKnn = { similitud: "coseno", vecinos: 50, encoger: 10, ...opciones };
    const n = r.columnas;
    // Pesos por usuario.
    const w = new Float64Array(r.valores.length);
    for (let u = 0; u < r.filas; u++) {
      const a = r.inicio[u]!;
      const b = r.inicio[u + 1]!;
      let media = 0;
      for (let p = a; p < b; p++) media += Math.log1p(r.valores[p]!);
      media /= Math.max(1, b - a);
      for (let p = a; p < b; p++) {
        const v = Math.log1p(r.valores[p]!);
        w[p] = o.similitud === "ajustada" ? v - media : v;
      }
    }
    const productos = new Float64Array(n * n);
    const comunes = new Float64Array(n * n);
    const normas = new Float64Array(n);
    for (let u = 0; u < r.filas; u++) {
      const a = r.inicio[u]!;
      const b = r.inicio[u + 1]!;
      for (let p = a; p < b; p++) {
        const i = r.indices[p]!;
        const wi = w[p]!;
        normas[i]! += wi * wi;
        for (let q = p + 1; q < b; q++) {
          const j = r.indices[q]!;
          const v = wi * w[q]!;
          productos[i * n + j]! += v;
          productos[j * n + i]! += v;
          comunes[i * n + j]! += 1;
          comunes[j * n + i]! += 1;
        }
      }
    }
    const vecinosDe = [];
    for (let j = 0; j < n; j++) {
      const candidatos: [number, number][] = [];
      for (let i = 0; i < n; i++) {
        if (i === j) continue;
        const c = comunes[j * n + i]!;
        if (c === 0) continue;
        const den = Math.sqrt(normas[i]! * normas[j]!);
        if (den === 0) continue;
        const s = (productos[j * n + i]! / den) * (c / (c + o.encoger));
        if (s > 0) candidatos.push([i, s]);
      }
      candidatos.sort((x, y) => y[1] - x[1] || x[0] - y[0]);
      const top = candidatos.slice(0, o.vecinos);
      vecinosDe.push({ items: Int32Array.from(top.map((t) => t[0])), sims: Float64Array.from(top.map((t) => t[1])) });
    }
    return new ItemKnn(r, vecinosDe, o.similitud);
  }

  puntuar(usuario: number, salida: Float64Array): void {
    salida.fill(0);
    for (const j of fila(this.historia, usuario).indices) {
      const v = this.vecinosDe[j]!;
      for (let k = 0; k < v.items.length; k++) salida[v.items[k]!]! += v.sims[k]!;
    }
  }

  explicar(usuario: number, item: number, cuantos = 3): Aporte[] {
    const aportes: Aporte[] = [];
    for (const j of fila(this.historia, usuario).indices) {
      const v = this.vecinosDe[j]!;
      const k = v.items.indexOf(item);
      if (k >= 0) aportes.push({ item: j, aporte: v.sims[k]! });
    }
    return aportes.sort((a, b) => b.aporte - a.aporte).slice(0, cuantos);
  }
}
