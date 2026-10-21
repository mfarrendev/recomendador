import { construir, fila, type Dispersa } from "./datos.ts";
import { Als } from "./modelos/als.ts";
import { Bpr } from "./modelos/bpr.ts";
import { ItemKnn, type Similitud } from "./modelos/knn.ts";
import type { Modelo } from "./modelos/modelo.ts";
import { Popularidad, PopularidadSegmento } from "./modelos/popularidad.ts";

/**
 * Guardado de un modelo entrenado como JSON. Se guarda también la historia
 * de entrenamiento (qué tocó cada usuario), porque recomendar necesita
 * excluir lo visto y kNN y ALS la usan para puntuar y explicar.
 */
export interface Guardado {
  formato: 1;
  tipo: "popularidad" | "segmento" | "knn" | "als" | "bpr";
  nUsuarios: number;
  nItems: number;
  historia: [number[], number[]][];
  nombresItems?: string[];
  datos: Record<string, unknown>;
}

const arr = (a: ArrayLike<number>) => Array.from(a);

export function guardar(modelo: Modelo, historia: Dispersa, nombresItems?: string[]): Guardado {
  const base = {
    formato: 1 as const,
    nUsuarios: historia.filas,
    nItems: historia.columnas,
    historia: Array.from({ length: historia.filas }, (_, u) => {
      const h = fila(historia, u);
      return [arr(h.indices), arr(h.valores)] as [number[], number[]];
    }),
    ...(nombresItems ? { nombresItems } : {}),
  };
  if (modelo instanceof Popularidad) return { ...base, tipo: "popularidad", datos: { conteo: arr(modelo.conteo) } };
  if (modelo instanceof PopularidadSegmento)
    return { ...base, tipo: "segmento", datos: { segmentos: arr(modelo.segmentos), tabla: modelo.tabla.map(arr) } };
  if (modelo instanceof ItemKnn)
    return {
      ...base,
      tipo: "knn",
      datos: { similitud: modelo.nombre.includes("ajustada") ? "ajustada" : "coseno", vecinos: modelo.vecinosDe.map((v) => [arr(v.items), arr(v.sims)]) },
    };
  if (modelo instanceof Als)
    return { ...base, tipo: "als", datos: { f: modelo.f, x: arr(modelo.x), y: arr(modelo.y), lambda: modelo.lambda, alfa: modelo.alfa } };
  if (modelo instanceof Bpr) return { ...base, tipo: "bpr", datos: { f: modelo.f, x: arr(modelo.x), y: arr(modelo.y), sesgo: arr(modelo.sesgo) } };
  throw new Error(`no sé guardar ${modelo.nombre}`);
}

export function cargar(g: Guardado): { modelo: Modelo; historia: Dispersa } {
  if (g.formato !== 1) throw new Error(`formato de modelo desconocido: ${String(g.formato)}`);
  const triples: [number, number, number][] = [];
  g.historia.forEach(([items, valores], u) => items.forEach((i, k) => triples.push([u, i, valores[k]!])));
  const historia = construir(g.nUsuarios, g.nItems, triples);
  const d = g.datos as Record<string, never>;
  const f64 = (a: number[]) => Float64Array.from(a);
  switch (g.tipo) {
    case "popularidad":
      return { modelo: new Popularidad(f64(d.conteo)), historia };
    case "segmento":
      return { modelo: new PopularidadSegmento(Int32Array.from(d.segmentos as number[]), (d.tabla as number[][]).map(f64)), historia };
    case "knn":
      return {
        modelo: new ItemKnn(
          historia,
          (d.vecinos as [number[], number[]][]).map(([it, s]) => ({ items: Int32Array.from(it), sims: f64(s) })),
          d.similitud as Similitud,
        ),
        historia,
      };
    case "als":
      return { modelo: new Als(d.f, f64(d.x), f64(d.y), g.nItems, historia, d.lambda, d.alfa), historia };
    case "bpr":
      return { modelo: new Bpr(d.f, f64(d.x), f64(d.y), f64(d.sesgo)), historia };
  }
}
