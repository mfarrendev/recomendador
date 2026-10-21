/*
 * Medición (2026-10-20): en src/particion.ts se reemplazó el arreglo
 * ordenado por tiempo por los eventos en el orden de entrada (la prueba los
 * baraja). Cayeron 2 de 3: sin futuro en entrenamiento, y empates en el
 * corte; el corredor ejecutó las 3. Revertido.
 */
import { describe, expect, it } from "vitest";
import { Azar } from "../src/azar.ts";
import type { Interaccion } from "../src/datos.ts";
import { particionTemporal } from "../src/particion.ts";
import { generar } from "../src/sintetico.ts";

function barajar<T>(xs: T[], semilla: number): T[] {
  const a = new Azar(semilla);
  const out = xs.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = a.entero(i + 1);
    [out[i], out[j]] = [out[j]!, out[i]!];
  }
  return out;
}

describe("partición temporal por usuario", () => {
  it("ningún evento de entrenamiento es posterior a uno de validación o prueba del mismo usuario", () => {
    const c = generar({ nUsuarios: 200, semilla: 5 });
    const p = particionTemporal(barajar(c.interacciones, 1));
    const maxEnt = new Map<number, number>();
    const minVal = new Map<number, number>();
    const minPru = new Map<number, number>();
    const actualizar = (m: Map<number, number>, x: Interaccion, f: (a: number, b: number) => number) =>
      m.set(x.usuario, m.has(x.usuario) ? f(m.get(x.usuario)!, x.tiempo) : x.tiempo);
    p.entrenamiento.forEach((x) => actualizar(maxEnt, x, Math.max));
    p.validacion.forEach((x) => actualizar(minVal, x, Math.min));
    p.prueba.forEach((x) => actualizar(minPru, x, Math.min));
    expect(p.prueba.length).toBeGreaterThan(0);
    for (const [u, t] of minPru) expect(t).toBeGreaterThan(maxEnt.get(u)!);
    for (const [u, t] of minVal) expect(t).toBeGreaterThan(maxEnt.get(u)!);
    const maxVal = new Map<number, number>();
    p.validacion.forEach((x) => actualizar(maxVal, x, Math.max));
    for (const [u, t] of minPru) if (maxVal.has(u)) expect(t).toBeGreaterThan(maxVal.get(u)!);
  });

  it("un empate en el corte no queda partido entre entrenamiento y prueba", () => {
    const ev: Interaccion[] = [];
    // 10 eventos; los últimos 4 comparten tiempo.
    for (let k = 0; k < 10; k++) ev.push({ usuario: 0, item: k, valor: 1, tiempo: k < 6 ? k : 6 });
    const p = particionTemporal(barajar(ev, 2), 0.2, 0.2, 3);
    const tEnt = Math.max(...p.entrenamiento.map((x) => x.tiempo));
    for (const x of [...p.validacion, ...p.prueba]) expect(x.tiempo).toBeGreaterThan(tEnt);
  });

  it("no deja en prueba un ítem que el usuario ya tocó antes", () => {
    const ev: Interaccion[] = [];
    for (let k = 0; k < 10; k++) ev.push({ usuario: 0, item: k % 4, valor: 1, tiempo: k });
    const p = particionTemporal(ev);
    const antes = new Set([...p.entrenamiento, ...p.validacion].map((x) => x.item));
    for (const x of p.prueba) expect(antes.has(x.item)).toBe(false);
  });
});
