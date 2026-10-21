import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import type { Conjunto, Interaccion } from "./datos.ts";

/**
 * Lector de MovieLens 100K (carpeta `ml-100k` tal como viene en el zip de
 * GroupLens). No se incluye en el repositorio: su licencia pide bajarlo del
 * sitio original.
 *
 * - `u.data`: usuario \t ítem \t nota \t marca de tiempo (ids desde 1).
 * - `u.item`: id | título | fecha | fecha video | url | 19 indicadoras de género.
 * - `u.user`: id | edad | sexo | ocupación | código postal.
 *
 * Se tratan como implícitas: una nota cualquiera es una interacción, y la
 * nota se conserva como `valor` (la confianza de ALS crece con ella). Con
 * `notaMinima` se puede quedar solo con lo que gustó (por ejemplo, >= 4).
 * Los géneros son los atributos de los ítems; la ocupación es el segmento.
 */
export function leerMovieLens(carpeta: string, notaMinima = 1): Conjunto {
  const ruta = (f: string) => join(carpeta, f);
  if (!existsSync(ruta("u.data"))) throw new Error(`no encuentro ${ruta("u.data")}; ¿bajaste ml-100k?`);

  const usuarios = new Map<number, number>();
  const items = new Map<number, number>();
  const denso = (m: Map<number, number>, id: number) => {
    let d = m.get(id);
    if (d === undefined) m.set(id, (d = m.size));
    return d;
  };

  // Primero los ítems de u.item, para que los ids densos sigan el archivo.
  const nombres: string[] = [];
  const generos: number[][] = [];
  if (existsSync(ruta("u.item"))) {
    // u.item viene en latin1.
    for (const linea of readFileSync(ruta("u.item"), "latin1").split("\n")) {
      if (!linea.trim()) continue;
      const c = linea.split("|");
      const d = denso(items, Number(c[0]));
      nombres[d] = c[1] ?? `película ${c[0]}`;
      generos[d] = c.slice(5, 24).map(Number);
    }
  }

  const interacciones: Interaccion[] = [];
  for (const linea of readFileSync(ruta("u.data"), "utf8").split("\n")) {
    if (!linea.trim()) continue;
    const [u, i, nota, t] = linea.split("\t").map(Number) as [number, number, number, number];
    if (nota < notaMinima) continue;
    interacciones.push({ usuario: denso(usuarios, u), item: denso(items, i), valor: nota, tiempo: t });
  }

  const conjunto: Conjunto = { nUsuarios: usuarios.size, nItems: items.size, interacciones };
  if (generos.length === items.size) {
    conjunto.atributos = generos.map((g) => Float64Array.from(g));
    conjunto.nombresItems = nombres;
  }
  if (existsSync(ruta("u.user"))) {
    const ocupaciones = new Map<string, number>();
    const seg = new Int32Array(usuarios.size);
    for (const linea of readFileSync(ruta("u.user"), "utf8").split("\n")) {
      if (!linea.trim()) continue;
      const c = linea.split("|");
      const d = usuarios.get(Number(c[0]));
      if (d === undefined) continue;
      const ocupacion = c[3] ?? "";
      if (!ocupaciones.has(ocupacion)) ocupaciones.set(ocupacion, ocupaciones.size);
      seg[d] = ocupaciones.get(ocupacion)!;
    }
    conjunto.segmentos = seg;
  }
  return conjunto;
}
