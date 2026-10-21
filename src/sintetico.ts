import { Azar } from "./azar.ts";
import type { Conjunto, Interaccion } from "./datos.ts";

export interface OpcionesSintetico {
  nUsuarios: number;
  nItems: number;
  grupos: number;
  /** Interacciones (eventos) promedio por usuario. */
  eventosPorUsuario: number;
  /** Probabilidad de que un evento caiga dentro del grupo de gusto del usuario. */
  fuerzaGusto: number;
  /** Exponente de Zipf de la popularidad: más alto, cola más larga. */
  zipf: number;
  /** Probabilidad de que el segmento observable coincida con el grupo real. */
  fidelidadSegmento: number;
  /** Ruido en los atributos de los ítems (desviación estándar). */
  ruidoAtributos: number;
  semilla: number;
}

export const POR_OMISION: OpcionesSintetico = {
  nUsuarios: 600,
  nItems: 400,
  grupos: 8,
  eventosPorUsuario: 40,
  fuerzaGusto: 0.8,
  zipf: 1.0,
  fidelidadSegmento: 0.5,
  ruidoAtributos: 0.6,
  semilla: 42,
};

/**
 * Datos con estructura latente conocida. Cada usuario y cada ítem pertenecen
 * a uno de `grupos` grupos de gusto. Un evento del usuario elige, con
 * probabilidad `fuerzaGusto`, un ítem de su propio grupo y si no uno
 * cualquiera; dentro de esa elección, los ítems pesan según una ley de Zipf,
 * así hay unos pocos éxitos y una cola larga. Como sabemos la verdad, una
 * prueba puede preguntar si el modelo la recuperó.
 *
 * Además se generan:
 * - un segmento observable por usuario que coincide con su grupo solo a veces
 *   (lo que tendría una empresa: región, plan, edad; correlacionado, no igual);
 * - atributos por ítem: la indicadora de su grupo más ruido gaussiano, y dos
 *   dimensiones de puro ruido (lo que serían metadatos imperfectos).
 */
export function generar(opciones: Partial<OpcionesSintetico> = {}): Conjunto {
  const o = { ...POR_OMISION, ...opciones };
  const azar = new Azar(o.semilla);

  const grupoUsuario = new Int32Array(o.nUsuarios);
  for (let u = 0; u < o.nUsuarios; u++) grupoUsuario[u] = azar.entero(o.grupos);
  const grupoItem = new Int32Array(o.nItems);
  for (let i = 0; i < o.nItems; i++) grupoItem[i] = i % o.grupos;

  // Popularidad de Zipf sobre un orden aleatorio de los ítems.
  const orden = Array.from({ length: o.nItems }, (_, i) => i);
  for (let i = orden.length - 1; i > 0; i--) {
    const j = azar.entero(i + 1);
    [orden[i], orden[j]] = [orden[j]!, orden[i]!];
  }
  const peso = new Float64Array(o.nItems);
  orden.forEach((item, rango) => (peso[item] = 1 / Math.pow(rango + 1, o.zipf)));

  const acumular = (items: number[]) => {
    const acc = new Float64Array(items.length);
    let s = 0;
    items.forEach((it, k) => (acc[k] = s += peso[it]!));
    return acc;
  };
  const todos = Array.from({ length: o.nItems }, (_, i) => i);
  const accTodos = acumular(todos);
  const delGrupo = Array.from({ length: o.grupos }, (_, g) => todos.filter((i) => grupoItem[i] === g));
  const accGrupo = delGrupo.map(acumular);

  const interacciones: Interaccion[] = [];
  for (let u = 0; u < o.nUsuarios; u++) {
    // Número de eventos variable: algunos usuarios son mucho más activos.
    const n = Math.max(5, Math.round(o.eventosPorUsuario * Math.exp(0.5 * azar.normal() - 0.125)));
    const g = grupoUsuario[u]!;
    // Cada usuario tiene su propia línea de tiempo con un desfase global.
    let t = Math.floor(azar.siguiente() * 1000);
    for (let e = 0; e < n; e++) {
      const item =
        azar.siguiente() < o.fuerzaGusto
          ? delGrupo[g]![azar.ponderado(accGrupo[g]!)]!
          : todos[azar.ponderado(accTodos)]!;
      t += 1 + azar.entero(10);
      interacciones.push({ usuario: u, item, valor: 1, tiempo: t });
    }
  }

  const segmentos = new Int32Array(o.nUsuarios);
  for (let u = 0; u < o.nUsuarios; u++) {
    segmentos[u] = azar.siguiente() < o.fidelidadSegmento ? grupoUsuario[u]! : azar.entero(o.grupos);
  }

  const atributos: Float64Array[] = [];
  for (let i = 0; i < o.nItems; i++) {
    const v = new Float64Array(o.grupos + 2);
    for (let d = 0; d < v.length; d++) v[d] = o.ruidoAtributos * azar.normal();
    v[grupoItem[i]!]! += 1;
    atributos.push(v);
  }

  return {
    nUsuarios: o.nUsuarios,
    nItems: o.nItems,
    interacciones,
    atributos,
    segmentos,
    nombresItems: todos.map((i) => `item-${i} (grupo ${grupoItem[i]})`),
    verdad: { grupoUsuario, grupoItem },
  };
}
