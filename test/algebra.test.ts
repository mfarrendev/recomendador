/*
 * Medición (2026-10-20): en src/algebra.ts se cambió, en la sustitución hacia
 * atrás, `l[k * n + i]` por `l[i * n + k]` (usar L en vez de Lᵀ).
 * Cayeron 2 de 4 pruebas (la de solución conocida y la de matriz aleatoria);
 * el corredor ejecutó las 4. Revertido.
 */
import { describe, expect, it } from "vitest";
import { cholesky, gram, resolverSPD } from "../src/algebra.ts";
import { Azar } from "../src/azar.ts";

describe("Cholesky", () => {
  it("resuelve un sistema con solución conocida", () => {
    // A = [[4,12,-16],[12,37,-43],[-16,-43,98]], L = [[2,0,0],[6,1,0],[-8,5,3]].
    const a = Float64Array.from([4, 12, -16, 12, 37, -43, -16, -43, 98]);
    expect(Array.from(cholesky(a, 3))).toEqual([2, 0, 0, 6, 1, 0, -8, 5, 3]);
    // x = (1, -2, 3) => b = A x
    const b = [4 * 1 + 12 * -2 + -16 * 3, 12 * 1 + 37 * -2 + -43 * 3, -16 * 1 + -43 * -2 + 98 * 3];
    const x = resolverSPD(a, 3, b);
    expect(x[0]).toBeCloseTo(1, 10);
    expect(x[1]).toBeCloseTo(-2, 10);
    expect(x[2]).toBeCloseTo(3, 10);
  });

  it("resuelve sistemas aleatorios 20 x 20 (residuo casi cero)", () => {
    const azar = new Azar(3);
    const n = 20;
    const m = new Float64Array(30 * n).map(() => azar.normal());
    const a = gram(m, 30, n);
    for (let i = 0; i < n; i++) a[i * n + i]! += 0.1;
    const xVerdad = Array.from({ length: n }, () => azar.normal());
    const b = new Float64Array(n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) b[i]! += a[i * n + j]! * xVerdad[j]!;
    const x = resolverSPD(a, n, b);
    for (let i = 0; i < n; i++) expect(x[i]).toBeCloseTo(xVerdad[i]!, 8);
  });

  it("rechaza una matriz que no es definida positiva", () => {
    expect(() => cholesky(Float64Array.from([1, 2, 2, 1]), 2)).toThrow(/definida positiva/);
  });

  it("gram calcula YᵀY simétrica", () => {
    const y = Float64Array.from([1, 2, 3, 4, 5, 6]); // 3 x 2
    expect(Array.from(gram(y, 3, 2))).toEqual([35, 44, 44, 56]);
  });
});
