/** Una interacción implícita: el usuario tocó el ítem `valor` veces en `tiempo`. */
export interface Interaccion {
  usuario: number;
  item: number;
  valor: number;
  tiempo: number;
}

/**
 * Un conjunto de datos. Los identificadores son enteros densos 0..n-1; los
 * lectores (sintético, MovieLens) se encargan de traducir.
 */
export interface Conjunto {
  nUsuarios: number;
  nItems: number;
  interacciones: Interaccion[];
  /** Vector de atributos por ítem (para arranque en frío). Opcional. */
  atributos?: Float64Array[];
  /** Segmento observable de cada usuario (región, edad...). Opcional. */
  segmentos?: Int32Array;
  /** Nombres legibles de los ítems. Opcional. */
  nombresItems?: string[];
  /** Verdad del generador sintético: grupo de gusto de usuarios e ítems. */
  verdad?: { grupoUsuario: Int32Array; grupoItem: Int32Array };
}

/**
 * Matriz dispersa en formato CSR. Se construye una por usuarios (filas =
 * usuarios) y su traspuesta por ítems; ALS necesita las dos.
 */
export interface Dispersa {
  filas: number;
  columnas: number;
  inicio: Int32Array; // largo filas + 1
  indices: Int32Array;
  valores: Float64Array;
}

/** Construye la matriz usuario x ítem sumando interacciones repetidas. */
export function matrizUsuarios(nUsuarios: number, nItems: number, inter: readonly Interaccion[]): Dispersa {
  return construir(nUsuarios, nItems, inter.map((x) => [x.usuario, x.item, x.valor] as const));
}

export function construir(
  filas: number,
  columnas: number,
  triples: ReadonlyArray<readonly [number, number, number]>,
): Dispersa {
  const porFila: Map<number, number>[] = Array.from({ length: filas }, () => new Map());
  for (const [f, c, v] of triples) {
    if (f < 0 || f >= filas || c < 0 || c >= columnas) throw new Error(`índice fuera de rango: (${f}, ${c})`);
    const m = porFila[f]!;
    m.set(c, (m.get(c) ?? 0) + v);
  }
  const inicio = new Int32Array(filas + 1);
  for (let f = 0; f < filas; f++) inicio[f + 1] = inicio[f]! + porFila[f]!.size;
  const nnz = inicio[filas]!;
  const indices = new Int32Array(nnz);
  const valores = new Float64Array(nnz);
  for (let f = 0; f < filas; f++) {
    const orden = [...porFila[f]!.entries()].sort((a, b) => a[0] - b[0]);
    let p = inicio[f]!;
    for (const [c, v] of orden) {
      indices[p] = c;
      valores[p] = v;
      p++;
    }
  }
  return { filas, columnas, inicio, indices, valores };
}

export function traspuesta(m: Dispersa): Dispersa {
  const triples: [number, number, number][] = [];
  for (let f = 0; f < m.filas; f++) {
    for (let p = m.inicio[f]!; p < m.inicio[f + 1]!; p++) triples.push([m.indices[p]!, f, m.valores[p]!]);
  }
  return construir(m.columnas, m.filas, triples);
}

/** Ítems de la fila `f` (por ejemplo, lo que vio un usuario). */
export function fila(m: Dispersa, f: number): { indices: Int32Array; valores: Float64Array } {
  const a = m.inicio[f]!;
  const b = m.inicio[f + 1]!;
  return { indices: m.indices.subarray(a, b), valores: m.valores.subarray(a, b) };
}

/** Cantidad de usuarios distintos que tocaron cada ítem. */
export function popularidad(m: Dispersa): Float64Array {
  const pop = new Float64Array(m.columnas);
  for (let p = 0; p < m.indices.length; p++) pop[m.indices[p]!]! += 1;
  return pop;
}
