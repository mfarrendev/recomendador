import { fila, popularidad, type Dispersa } from "../datos.ts";
import type { Modelo } from "./modelo.ts";

/** Recomienda lo más popular. El piso que cualquier modelo tiene que superar. */
export class Popularidad implements Modelo {
  readonly nombre = "popularidad";
  readonly conteo: Float64Array;

  constructor(conteo: Float64Array) {
    this.conteo = conteo;
  }

  static entrenar(r: Dispersa): Popularidad {
    return new Popularidad(popularidad(r));
  }

  puntuar(_usuario: number, salida: Float64Array): void {
    salida.set(this.conteo);
  }
}

/**
 * Popularidad dentro del segmento observable del usuario (región, plan,
 * ocupación). Para segmentos chicos la cuenta es ruidosa, así que se encoge
 * hacia la popularidad global con un peso `m` en usuarios equivalentes:
 * puntaje = (cuenta_en_segmento + m · tasa_global) / (tamaño_segmento + m).
 */
export class PopularidadSegmento implements Modelo {
  readonly nombre = "popularidad por segmento";
  readonly segmentos: Int32Array;
  /** Fila por segmento: puntaje ya encogido de cada ítem. */
  readonly tabla: Float64Array[];

  constructor(segmentos: Int32Array, tabla: Float64Array[]) {
    this.segmentos = segmentos;
    this.tabla = tabla;
  }

  static entrenar(r: Dispersa, segmentos: Int32Array, m = 10): PopularidadSegmento {
    const nSeg = Math.max(0, ...segmentos) + 1;
    const global = popularidad(r);
    const nUsuarios = r.filas;
    const cuenta = Array.from({ length: nSeg }, () => new Float64Array(r.columnas));
    const tam = new Float64Array(nSeg);
    for (let u = 0; u < r.filas; u++) {
      const s = segmentos[u]!;
      tam[s]! += 1;
      for (const i of fila(r, u).indices) cuenta[s]![i]! += 1;
    }
    const tabla = cuenta.map((c, s) => {
      const t = new Float64Array(r.columnas);
      for (let i = 0; i < r.columnas; i++) t[i] = (c[i]! + (m * global[i]!) / nUsuarios) / (tam[s]! + m);
      return t;
    });
    return new PopularidadSegmento(segmentos, tabla);
  }

  puntuar(usuario: number, salida: Float64Array): void {
    salida.set(this.tabla[this.segmentos[usuario]!]!);
  }
}
