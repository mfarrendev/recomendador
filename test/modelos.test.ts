/*
 * Mediciones (2026-10-20), cada una aplicada a mano por un arnés, corrida y
 * revertida; el arnés exige que el corredor haya ejecutado las 7 pruebas:
 * - ALS: en pasoAls (src/modelos/als.ts), `c * fijo[o + d]` cambiado por
 *   `fijo[o + d]` (lado derecho sin confianza: el medio paso deja de ser el
 *   mínimo exacto). Cayeron 2 de 7: la pérdida monótona y la explicación
 *   que suma el puntaje.
 * - BPR: sigmoide invertida, `1 / (1 + Math.exp(-diff))`. Cayó 1 de 7: ALS
 *   y BPR superan a popularidad. (Invertir solo el signo del gradiente de
 *   y_j NO cayó: 0 de 7; esa mutación degrada BPR sin bajarlo de 1,3 veces
 *   popularidad. Queda anotado como punto ciego.)
 * - kNN: `if (s > 0)` cambiado por `if (s < 0)` en ItemKnn.entrenar.
 *   Cayeron 2 de 7: mismo grupo, y kNN/ALS explican con ítems vistos.
 * - recomendar: se quitó `if (excluir.has(i)) continue;`. Cayeron 2 de 7:
 *   ningún ítem visto, y ALS/BPR contra popularidad.
 */
import { describe, expect, it } from "vitest";
import { fila } from "../src/datos.ts";
import { evaluar } from "../src/evaluar.ts";
import { Als } from "../src/modelos/als.ts";
import { Bpr } from "../src/modelos/bpr.ts";
import { ItemKnn } from "../src/modelos/knn.ts";
import { recomendar, type Modelo } from "../src/modelos/modelo.ts";
import { Popularidad, PopularidadSegmento } from "../src/modelos/popularidad.ts";
import { Contenido } from "../src/modelos/frio.ts";
import { estructuraFuerte } from "./comun.ts";

describe("ALS implícito", () => {
  it("la pérdida no sube en ningún medio paso", () => {
    const { r } = estructuraFuerte();
    const perdidas: number[] = [];
    Als.entrenar(r, { factores: 8, lambda: 0.5, alfa: 15, iteraciones: 8, alPaso: (p) => perdidas.push(p) });
    expect(perdidas).toHaveLength(16);
    for (let k = 1; k < perdidas.length; k++) {
      // Tolerancia relativa por redondeo de punto flotante.
      expect(perdidas[k]!).toBeLessThanOrEqual(perdidas[k - 1]! * (1 + 1e-9));
    }
    expect(perdidas.at(-1)!).toBeLessThan(perdidas[0]! * 0.9);
  });

  it("los aportes de la explicación suman exactamente el puntaje", () => {
    const { r } = estructuraFuerte();
    const als = Als.entrenar(r, { factores: 8, lambda: 1, alfa: 10, iteraciones: 5 });
    const puntajes = new Float64Array(r.columnas);
    for (const u of [0, 17, 123]) {
      als.puntuar(u, puntajes);
      for (const i of [3, 50, 199]) {
        const suma = als.aportes(u, i).reduce((s, a) => s + a.aporte, 0);
        expect(suma).toBeCloseTo(puntajes[i]!, 8);
      }
    }
  });
});

describe("contra la verdad del generador", () => {
  it("ALS y BPR superan a popularidad en NDCG@10 cuando la estructura es fuerte", () => {
    const { p, r } = estructuraFuerte();
    const pop = evaluar(Popularidad.entrenar(r), r, p.prueba).ndcg;
    const als = evaluar(Als.entrenar(r, { factores: 16, lambda: 1, alfa: 10, iteraciones: 10 }), r, p.prueba).ndcg;
    const bpr = evaluar(Bpr.entrenar(r, { factores: 16, regularizacion: 0.01, epocas: 30 }), r, p.prueba).ndcg;
    expect(als).toBeGreaterThan(pop * 1.3);
    expect(bpr).toBeGreaterThan(pop * 1.3);
  });

  it("item-kNN recomienda sobre todo ítems del grupo de gusto del usuario", () => {
    const { conjunto, r } = estructuraFuerte();
    const knn = ItemKnn.entrenar(r, { vecinos: 30 });
    const { grupoUsuario, grupoItem } = conjunto.verdad!;
    let mismos = 0;
    let total = 0;
    for (let u = 0; u < conjunto.nUsuarios; u++) {
      for (const i of recomendar(knn, u, 10, fila(r, u).indices, r.columnas)) {
        total++;
        if (grupoItem[i] === grupoUsuario[u]) mismos++;
      }
    }
    // Al azar sería 1/5; popularidad, cerca de eso.
    expect(total).toBe(conjunto.nUsuarios * 10);
    expect(mismos / total).toBeGreaterThan(0.85);
  });

  it("kNN y ALS explican con ítems que el usuario sí vio", () => {
    const { r } = estructuraFuerte();
    const modelos = [ItemKnn.entrenar(r), Als.entrenar(r, { factores: 8, iteraciones: 5 })];
    for (const m of modelos) {
      for (const u of [1, 2, 3]) {
        const vistos = new Set(fila(r, u).indices);
        const [primero] = recomendar(m, u, 1, vistos, r.columnas);
        const ex = m.explicar(u, primero!, 3);
        expect(ex.length).toBeGreaterThan(0);
        for (const a of ex) expect(vistos.has(a.item)).toBe(true);
        for (let k = 1; k < ex.length; k++) expect(ex[k]!.aporte).toBeLessThanOrEqual(ex[k - 1]!.aporte);
      }
    }
  });
});

describe("recomendar", () => {
  it("ningún modelo devuelve un ítem ya visto", () => {
    const { conjunto, r } = estructuraFuerte();
    const modelos: Modelo[] = [
      Popularidad.entrenar(r),
      PopularidadSegmento.entrenar(r, conjunto.segmentos!),
      ItemKnn.entrenar(r),
      ItemKnn.entrenar(r, { similitud: "ajustada" }),
      Als.entrenar(r, { factores: 8, iteraciones: 3 }),
      Bpr.entrenar(r, { factores: 8, epocas: 3 }),
      new Contenido(conjunto.atributos!, r),
    ];
    for (const m of modelos) {
      for (let u = 0; u < conjunto.nUsuarios; u += 7) {
        const vistos = new Set(fila(r, u).indices);
        const recs = recomendar(m, u, 20, vistos, r.columnas);
        expect(recs).toHaveLength(20);
        expect(new Set(recs).size).toBe(20);
        for (const i of recs) expect(vistos.has(i), `${m.nombre} recomendó ${i} a ${u}`).toBe(false);
      }
    }
  });

  it("devuelve en orden de puntaje descendente", () => {
    const { r } = estructuraFuerte();
    const m = Bpr.entrenar(r, { factores: 8, epocas: 3 });
    const s = new Float64Array(r.columnas);
    m.puntuar(5, s);
    const recs = recomendar(m, 5, 30, fila(r, 5).indices, r.columnas);
    for (let k = 1; k < recs.length; k++) expect(s[recs[k]!]!).toBeLessThanOrEqual(s[recs[k - 1]!]!);
  });
});
