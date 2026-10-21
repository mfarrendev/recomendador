import { Azar } from "./azar.ts";
import { matrizUsuarios, type Conjunto, type Dispersa, type Interaccion } from "./datos.ts";
import { evaluar, type Resultado } from "./evaluar.ts";
import { Als } from "./modelos/als.ts";
import { Bpr } from "./modelos/bpr.ts";
import { Contenido, Hibrido } from "./modelos/frio.ts";
import { ItemKnn } from "./modelos/knn.ts";
import type { Modelo } from "./modelos/modelo.ts";
import { Popularidad, PopularidadSegmento } from "./modelos/popularidad.ts";
import { particionTemporal } from "./particion.ts";

export type Hiper = Record<string, number | string>;

export interface Metodo {
  nombre: string;
  /** Grilla de hiperparámetros; un solo elemento vacío si no tiene. */
  grilla: Hiper[];
  entrenar(r: Dispersa, h: Hiper, conjunto: Conjunto): Modelo;
}

/** Producto cartesiano de listas de valores. */
export function grilla(ejes: Record<string, (number | string)[]>): Hiper[] {
  let out: Hiper[] = [{}];
  for (const [clave, valores] of Object.entries(ejes)) out = out.flatMap((h) => valores.map((v) => ({ ...h, [clave]: v })));
  return out;
}

export function metodos(conjunto: Conjunto, rapido = false): Metodo[] {
  const lista: Metodo[] = [{ nombre: "popularidad", grilla: [{}], entrenar: (r) => Popularidad.entrenar(r) }];
  if (conjunto.segmentos) {
    const seg = conjunto.segmentos;
    lista.push({
      nombre: "popularidad por segmento",
      grilla: grilla({ m: rapido ? [10] : [1, 10, 100] }),
      entrenar: (r, h) => PopularidadSegmento.entrenar(r, seg, Number(h.m)),
    });
  }
  lista.push(
    {
      nombre: "item-kNN coseno",
      grilla: grilla({ vecinos: rapido ? [50] : [20, 50, 200], encoger: rapido ? [10] : [0, 10] }),
      entrenar: (r, h) => ItemKnn.entrenar(r, { similitud: "coseno", vecinos: Number(h.vecinos), encoger: Number(h.encoger) }),
    },
    {
      nombre: "item-kNN ajustada",
      grilla: grilla({ vecinos: rapido ? [50] : [20, 50, 200], encoger: rapido ? [10] : [0, 10] }),
      entrenar: (r, h) => ItemKnn.entrenar(r, { similitud: "ajustada", vecinos: Number(h.vecinos), encoger: Number(h.encoger) }),
    },
    {
      nombre: "ALS implícito",
      grilla: grilla({ factores: rapido ? [16] : [16, 32], lambda: rapido ? [1] : [0.1, 1, 10], alfa: rapido ? [10] : [1, 10, 40] }),
      entrenar: (r, h) =>
        Als.entrenar(r, { factores: Number(h.factores), lambda: Number(h.lambda), alfa: Number(h.alfa), iteraciones: rapido ? 8 : 12 }),
    },
    {
      nombre: "BPR",
      grilla: grilla({ factores: rapido ? [16] : [16, 32], regularizacion: rapido ? [0.01] : [0.001, 0.01, 0.05] }),
      entrenar: (r, h) =>
        Bpr.entrenar(r, { factores: Number(h.factores), regularizacion: Number(h.regularizacion), epocas: rapido ? 20 : 40 }),
    },
  );
  return lista;
}

export interface Fila {
  metodo: string;
  mejor: Hiper;
  validacion: number;
  prueba: Resultado;
  segundos: number;
}

/**
 * Protocolo:
 * 1. Partición temporal por usuario: entrenamiento | validación | prueba.
 * 2. Para cada método, cada punto de su grilla se entrena con
 *    entrenamiento y se mide NDCG@k en validación. Prueba no se mira.
 * 3. Con el mejor punto se reentrena con entrenamiento + validación y se
 *    mide una sola vez en prueba. Así el número de prueba no está elegido
 *    sobre prueba.
 */
export function comparar(conjunto: Conjunto, k = 10, rapido = false, alProgreso?: (m: string) => void): Fila[] {
  const p = particionTemporal(conjunto.interacciones);
  const rEnt = matrizUsuarios(conjunto.nUsuarios, conjunto.nItems, p.entrenamiento);
  const rFinal = matrizUsuarios(conjunto.nUsuarios, conjunto.nItems, [...p.entrenamiento, ...p.validacion]);
  const filas: Fila[] = [];
  for (const m of metodos(conjunto, rapido)) {
    alProgreso?.(m.nombre);
    const t0 = performance.now();
    let mejor: Hiper = m.grilla[0]!;
    let mejorVal = -1;
    for (const h of m.grilla) {
      const v = evaluar(m.entrenar(rEnt, h, conjunto), rEnt, p.validacion, k).ndcg;
      if (v > mejorVal) [mejorVal, mejor] = [v, h];
    }
    const prueba = evaluar(m.entrenar(rFinal, mejor, conjunto), rFinal, p.prueba, k);
    filas.push({ metodo: m.nombre, mejor, validacion: mejorVal, prueba, segundos: (performance.now() - t0) / 1000 });
  }
  return filas;
}

export interface FilaFrio {
  metodo: string;
  resultado: Resultado;
}

/**
 * Arranque en frío de ítems. Se elige al azar una fracción de ítems que se
 * declaran "nuevos": se borran TODAS sus interacciones de entrenamiento y
 * validación. Después se mide, sobre las interacciones de prueba con esos
 * ítems, qué tan bien cada método los ordena entre sí (los candidatos son
 * solo los ítems nuevos: la pregunta es "¿a quién le muestro este estreno?").
 */
export function arranqueEnFrio(conjunto: Conjunto, k = 10, fraccion = 0.1, semilla = 7): FilaFrio[] {
  if (!conjunto.atributos) throw new Error("el arranque en frío necesita atributos de ítems");
  const azar = new Azar(semilla);
  const frios = new Set<number>();
  while (frios.size < Math.max(1, Math.round(conjunto.nItems * fraccion))) frios.add(azar.entero(conjunto.nItems));
  const p = particionTemporal(conjunto.interacciones);
  const pasado: Interaccion[] = [...p.entrenamiento, ...p.validacion].filter((x) => !frios.has(x.item));
  const r = matrizUsuarios(conjunto.nUsuarios, conjunto.nItems, pasado);
  const als = Als.entrenar(r, { factores: 16, lambda: 1, alfa: 10, iteraciones: 12 });
  const azarModelo: Modelo = {
    nombre: "al azar",
    puntuar: (_u, s) => {
      for (let i = 0; i < s.length; i++) s[i] = azar.siguiente();
    },
  };
  const modelos: Modelo[] = [azarModelo, Popularidad.entrenar(r), als, new Contenido(conjunto.atributos, r), new Hibrido(als, conjunto.atributos, frios)];
  return modelos.map((m) => ({ metodo: m.nombre, resultado: evaluar(m, r, p.prueba, k, frios) }));
}

const f3 = (x: number) => x.toFixed(3);

export function tablaMarkdown(filas: Fila[], k: number): string {
  const enc = `| método | P@${k} | R@${k} | NDCG@${k} | MAP@${k} | cobertura | novedad (bits) | pop. media | NDCG@${k} val. | hiperparámetros | s |`;
  const sep = "|" + "---|".repeat(11);
  const cuerpo = filas.map((f) => {
    const r = f.prueba;
    const h = Object.entries(f.mejor).map(([a, b]) => `${a}=${b}`).join(", ") || "-";
    return `| ${f.metodo} | ${f3(r.precision)} | ${f3(r.recall)} | ${f3(r.ndcg)} | ${f3(r.map)} | ${f3(r.cobertura)} | ${r.novedad.toFixed(2)} | ${f3(r.popularidadMedia)} | ${f3(f.validacion)} | ${h} | ${f.segundos.toFixed(1)} |`;
  });
  return [enc, sep, ...cuerpo].join("\n");
}

export function tablaFrio(filas: FilaFrio[], k: number): string {
  const enc = `| método | P@${k} | R@${k} | NDCG@${k} | cobertura de los nuevos |`;
  const cuerpo = filas.map((f) => `| ${f.metodo} | ${f3(f.resultado.precision)} | ${f3(f.resultado.recall)} | ${f3(f.resultado.ndcg)} | ${f3(f.resultado.cobertura)} |`);
  return [enc, "|---|---|---|---|---|", ...cuerpo].join("\n");
}
