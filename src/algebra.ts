/**
 * Álgebra lineal mínima sobre matrices densas cuadradas guardadas por filas
 * en un Float64Array (a[i * n + j]). Es lo que ALS necesita: sistemas f x f
 * simétricos definidos positivos, con f el número de factores (decenas).
 */

/**
 * Factorización de Cholesky A = L Lᵀ. Devuelve L (triangular inferior) en una
 * matriz nueva. Lanza si A no es definida positiva: en ALS eso indicaría que
 * falta regularización, y es mejor saberlo que resolver basura.
 */
export function cholesky(a: Float64Array, n: number): Float64Array {
  const l = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = a[i * n + j]!;
      for (let k = 0; k < j; k++) s -= l[i * n + k]! * l[j * n + k]!;
      if (i === j) {
        if (!(s > 0)) throw new Error(`la matriz no es definida positiva (pivote ${s} en ${i})`);
        l[i * n + i] = Math.sqrt(s);
      } else {
        l[i * n + j] = s / l[j * n + j]!;
      }
    }
  }
  return l;
}

/** Resuelve L Lᵀ x = b con sustitución hacia adelante y hacia atrás. */
export function resolverCholesky(l: Float64Array, n: number, b: ArrayLike<number>): Float64Array {
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i]!;
    for (let k = 0; k < i; k++) s -= l[i * n + k]! * y[k]!;
    y[i] = s / l[i * n + i]!;
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i]!;
    for (let k = i + 1; k < n; k++) s -= l[k * n + i]! * x[k]!;
    x[i] = s / l[i * n + i]!;
  }
  return x;
}

/** Atajo: resuelve A x = b con A simétrica definida positiva. */
export function resolverSPD(a: Float64Array, n: number, b: ArrayLike<number>): Float64Array {
  return resolverCholesky(cholesky(a, n), n, b);
}

export function punto(a: ArrayLike<number>, ao: number, b: ArrayLike<number>, bo: number, n: number): number {
  let s = 0;
  for (let k = 0; k < n; k++) s += a[ao + k]! * b[bo + k]!;
  return s;
}

/** Gram YᵀY de una matriz de m filas y f columnas (por filas). */
export function gram(y: Float64Array, m: number, f: number): Float64Array {
  const g = new Float64Array(f * f);
  for (let r = 0; r < m; r++) {
    const o = r * f;
    for (let i = 0; i < f; i++) {
      const yi = y[o + i]!;
      if (yi === 0) continue;
      for (let j = i; j < f; j++) g[i * f + j]! += yi * y[o + j]!;
    }
  }
  for (let i = 0; i < f; i++) for (let j = 0; j < i; j++) g[i * f + j] = g[j * f + i]!;
  return g;
}
