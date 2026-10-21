/**
 * Métricas de ranking con relevancia binaria: un ítem es relevante si el
 * usuario lo tocó en el período que se evalúa.
 */

export function precisionEnK(recs: readonly number[], relevantes: ReadonlySet<number>, k: number): number {
  let aciertos = 0;
  for (const i of recs.slice(0, k)) if (relevantes.has(i)) aciertos++;
  return aciertos / k;
}

export function recallEnK(recs: readonly number[], relevantes: ReadonlySet<number>, k: number): number {
  if (relevantes.size === 0) return 0;
  let aciertos = 0;
  for (const i of recs.slice(0, k)) if (relevantes.has(i)) aciertos++;
  return aciertos / relevantes.size;
}

/** NDCG@k: DCG con descuento log2(posición + 1), dividido por el DCG ideal. */
export function ndcgEnK(recs: readonly number[], relevantes: ReadonlySet<number>, k: number): number {
  let dcg = 0;
  recs.slice(0, k).forEach((i, pos) => {
    if (relevantes.has(i)) dcg += 1 / Math.log2(pos + 2);
  });
  let ideal = 0;
  for (let pos = 0; pos < Math.min(k, relevantes.size); pos++) ideal += 1 / Math.log2(pos + 2);
  return ideal === 0 ? 0 : dcg / ideal;
}

/**
 * Precisión promedio en k (AP@k), normalizada por min(|relevantes|, k): así
 * un usuario con 30 relevantes no queda condenado a un AP bajo por un tope
 * de 10 posiciones. MAP es el promedio de esto entre usuarios.
 */
export function apEnK(recs: readonly number[], relevantes: ReadonlySet<number>, k: number): number {
  if (relevantes.size === 0) return 0;
  let aciertos = 0;
  let suma = 0;
  recs.slice(0, k).forEach((i, pos) => {
    if (relevantes.has(i)) {
      aciertos++;
      suma += aciertos / (pos + 1);
    }
  });
  return suma / Math.min(relevantes.size, k);
}
