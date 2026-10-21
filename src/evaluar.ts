import { fila, popularidad, type Dispersa, type Interaccion } from "./datos.ts";
import { apEnK, ndcgEnK, precisionEnK, recallEnK } from "./metricas.ts";
import { recomendar, type Modelo } from "./modelos/modelo.ts";

export interface Resultado {
  precision: number;
  recall: number;
  ndcg: number;
  map: number;
  /** Fracción del catálogo que aparece en alguna lista. */
  cobertura: number;
  /** Autoinformación media de lo recomendado: −log2(fracción de usuarios que lo tocó). */
  novedad: number;
  /** Fracción media de usuarios que tocó cada ítem recomendado (sesgo de popularidad). */
  popularidadMedia: number;
  usuarios: number;
}

/**
 * Evalúa `modelo` contra `objetivo` (validación o prueba). Las listas
 * excluyen lo visto en `entrenamiento`; si se dan `candidatos`, solo se
 * ordena entre ellos (lo usa el experimento de arranque en frío).
 */
export function evaluar(
  modelo: Modelo,
  entrenamiento: Dispersa,
  objetivo: readonly Interaccion[],
  k = 10,
  candidatos?: ReadonlySet<number>,
): Resultado {
  const nItems = entrenamiento.columnas;
  const nUsuarios = entrenamiento.filas;
  const pop = popularidad(entrenamiento);
  const relevantes = new Map<number, Set<number>>();
  for (const x of objetivo) {
    if (candidatos && !candidatos.has(x.item)) continue;
    let s = relevantes.get(x.usuario);
    if (!s) relevantes.set(x.usuario, (s = new Set()));
    s.add(x.item);
  }
  const r: Resultado = { precision: 0, recall: 0, ndcg: 0, map: 0, cobertura: 0, novedad: 0, popularidadMedia: 0, usuarios: 0 };
  const recomendados = new Set<number>();
  let nRecs = 0;
  for (const [u, rel] of relevantes) {
    const recs = recomendar(modelo, u, k, fila(entrenamiento, u).indices, nItems, candidatos);
    r.precision += precisionEnK(recs, rel, k);
    r.recall += recallEnK(recs, rel, k);
    r.ndcg += ndcgEnK(recs, rel, k);
    r.map += apEnK(recs, rel, k);
    for (const i of recs) {
      recomendados.add(i);
      const frac = (pop[i]! + 1) / (nUsuarios + 1);
      r.novedad += -Math.log2(frac);
      r.popularidadMedia += pop[i]! / nUsuarios;
      nRecs++;
    }
    r.usuarios++;
  }
  const n = Math.max(1, r.usuarios);
  r.precision /= n;
  r.recall /= n;
  r.ndcg /= n;
  r.map /= n;
  r.novedad /= Math.max(1, nRecs);
  r.popularidadMedia /= Math.max(1, nRecs);
  r.cobertura = recomendados.size / (candidatos ? candidatos.size : nItems);
  return r;
}
