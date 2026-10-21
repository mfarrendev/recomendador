/*
 * Medición (2026-10-20): en src/metricas.ts, ndcgEnK con `Math.log2(pos + 1)`
 * en vez de `pos + 2` (descuento corrido en uno). Cayó 1 de 3 (la de NDCG);
 * el corredor ejecutó las 3. Revertido.
 */
import { describe, expect, it } from "vitest";
import { apEnK, ndcgEnK, precisionEnK, recallEnK } from "../src/metricas.ts";

describe("métricas de ranking", () => {
  const rel = new Set([1, 3, 9]);
  const recs = [1, 2, 3, 4, 5];

  it("precisión y recall", () => {
    expect(precisionEnK(recs, rel, 5)).toBeCloseTo(2 / 5);
    expect(recallEnK(recs, rel, 5)).toBeCloseTo(2 / 3);
  });

  it("NDCG calculado a mano", () => {
    const dcg = 1 / Math.log2(2) + 1 / Math.log2(4);
    const ideal = 1 / Math.log2(2) + 1 / Math.log2(3) + 1 / Math.log2(4);
    expect(ndcgEnK(recs, rel, 5)).toBeCloseTo(dcg / ideal, 12);
    expect(ndcgEnK([1, 3, 9], rel, 3)).toBeCloseTo(1, 12);
    expect(ndcgEnK([7, 8], rel, 2)).toBe(0);
  });

  it("AP normalizado por min(|rel|, k)", () => {
    // aciertos en posiciones 1 y 3: (1/1 + 2/3) / 3
    expect(apEnK(recs, rel, 5)).toBeCloseTo((1 + 2 / 3) / 3, 12);
    // con k = 2 solo cuenta la posición 1, normalizado por 2
    expect(apEnK(recs, rel, 2)).toBeCloseTo(1 / 2, 12);
  });
});
