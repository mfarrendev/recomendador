/*
 * Medición (2026-10-20): en src/movielens.ts se cambió `if (nota < notaMinima)`
 * por `if (nota <= notaMinima)`. Cayó 1 de 1; el corredor ejecutó 1. Revertido.
 */
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { leerMovieLens } from "../src/movielens.ts";

describe("lector de MovieLens 100K", () => {
  it("lee u.data, u.item y u.user con ids densos, géneros y ocupación", () => {
    const d = mkdtempSync(join(tmpdir(), "ml-"));
    try {
      writeFileSync(join(d, "u.data"), "196\t242\t3\t881250949\n186\t302\t3\t891717742\n196\t302\t5\t881250950\n22\t242\t1\t878887116\n");
      const g = (k: number) => Array.from({ length: 19 }, (_, i) => (i === k ? 1 : 0)).join("|");
      writeFileSync(join(d, "u.item"), `242|Kolya (1996)|24-Jan-1997||http://x|${g(5)}\n302|L.A. Confidential (1997)|01-Jan-1997||http://y|${g(7)}\n`);
      writeFileSync(join(d, "u.user"), "22|25|M|writer|40206\n186|39|F|executive|00000\n196|49|M|writer|55105\n");
      const c = leerMovieLens(d, 3);
      expect(c.nItems).toBe(2);
      expect(c.nUsuarios).toBe(2); // el usuario 22 solo tiene una nota 1, bajo el mínimo
      expect(c.interacciones).toHaveLength(3);
      expect(c.nombresItems).toEqual(["Kolya (1996)", "L.A. Confidential (1997)"]);
      expect(c.atributos![1]![7]).toBe(1);
      expect(c.interacciones[0]).toEqual({ usuario: 0, item: 0, valor: 3, tiempo: 881250949 });
      // 196 y 186 tienen ocupaciones distintas
      expect(c.segmentos![0]).not.toBe(c.segmentos![1]);
    } finally {
      rmSync(d, { recursive: true, force: true });
    }
  });
});
