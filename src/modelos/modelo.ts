/** Lo que un modelo entrenado sabe hacer. */
export interface Modelo {
  readonly nombre: string;
  /** Escribe en `salida` (largo nItems) el puntaje de cada ítem para `usuario`. */
  puntuar(usuario: number, salida: Float64Array): void;
}

/** Un ítem que explica una recomendación, con cuánto aportó al puntaje. */
export interface Aporte {
  item: number;
  aporte: number;
}

export interface Explicable {
  /** "Porque viste X": los ítems de la historia que más empujan `item`. */
  explicar(usuario: number, item: number, cuantos?: number): Aporte[];
}

export function esExplicable(m: Modelo): m is Modelo & Explicable {
  return typeof (m as Partial<Explicable>).explicar === "function";
}

/**
 * Los k ítems de mayor puntaje que el usuario NO ha visto. Los vistos se
 * excluyen siempre: recomendar lo que ya consumió es un acierto falso en la
 * evaluación y un desperdicio de espacio en el producto.
 */
export function recomendar(
  modelo: Modelo,
  usuario: number,
  k: number,
  vistos: ReadonlySet<number> | ArrayLike<number>,
  nItems: number,
  candidatos?: ReadonlySet<number>,
): number[] {
  const puntaje = new Float64Array(nItems);
  modelo.puntuar(usuario, puntaje);
  const excluir = vistos instanceof Set ? vistos : new Set(Array.from(vistos as ArrayLike<number>));
  // Selección parcial: mantener un arreglo ordenado de a lo más k.
  const top: number[] = [];
  for (let i = 0; i < nItems; i++) {
    if (excluir.has(i)) continue;
    if (candidatos && !candidatos.has(i)) continue;
    const s = puntaje[i]!;
    if (top.length === k && s <= puntaje[top[k - 1]!]!) continue;
    let pos = top.length;
    while (pos > 0 && puntaje[top[pos - 1]!]! < s) pos--;
    top.splice(pos, 0, i);
    if (top.length > k) top.pop();
  }
  return top;
}
