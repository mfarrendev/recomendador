/*
 * Medición (2026-10-20): en Hibrido (src/modelos/frio.ts) se anuló la
 * asignación del factor proyectado a los ítems fríos (quedan con el y de
 * ALS, que es cero). Cayó 1 de 2: el híbrido supera a ALS en ítems nuevos; el
 * corredor ejecutó las 2. Revertido.
 */
import { describe, expect, it } from "vitest";
import { fila, matrizUsuarios } from "../src/datos.ts";
import { arranqueEnFrio } from "../src/experimentos.ts";
import { Als } from "../src/modelos/als.ts";
import { ItemKnn } from "../src/modelos/knn.ts";
import { recomendar } from "../src/modelos/modelo.ts";
import { PopularidadSegmento } from "../src/modelos/popularidad.ts";
import { Bpr } from "../src/modelos/bpr.ts";
import { cargar, guardar } from "../src/persistencia.ts";
import { generar } from "../src/sintetico.ts";

describe("arranque en frío", () => {
  it("con ítems sin historia, contenido e híbrido superan a ALS puro y a popularidad", () => {
    const c = generar({ nUsuarios: 400, nItems: 300, grupos: 6, fuerzaGusto: 0.85, semilla: 21 });
    const filas = new Map(arranqueEnFrio(c, 10, 0.15).map((f) => [f.metodo, f.resultado.ndcg]));
    const als = filas.get("ALS implícito")!;
    const pop = filas.get("popularidad")!;
    expect(filas.get("contenido")!).toBeGreaterThan(Math.max(als, pop) * 1.5);
    expect(filas.get("híbrido (ALS + atributos)")!).toBeGreaterThan(Math.max(als, pop) * 1.5);
  });
});

describe("guardado", () => {
  it("un modelo cargado recomienda y explica lo mismo que el original", () => {
    const c = generar({ nUsuarios: 120, nItems: 90, semilla: 4 });
    const r = matrizUsuarios(c.nUsuarios, c.nItems, c.interacciones);
    const modelos = [
      Als.entrenar(r, { factores: 6, iteraciones: 3 }),
      ItemKnn.entrenar(r),
      Bpr.entrenar(r, { factores: 6, epocas: 2 }),
      PopularidadSegmento.entrenar(r, c.segmentos!),
    ];
    for (const m of modelos) {
      const g = JSON.parse(JSON.stringify(guardar(m, r)));
      const { modelo, historia } = cargar(g);
      for (const u of [0, 50, 119]) {
        const a = recomendar(m, u, 10, fila(r, u).indices, c.nItems);
        const b = recomendar(modelo, u, 10, fila(historia, u).indices, c.nItems);
        expect(b).toEqual(a);
      }
    }
  });
});
