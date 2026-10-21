import { matrizUsuarios } from "../src/datos.ts";
import { particionTemporal } from "../src/particion.ts";
import { generar } from "../src/sintetico.ts";

/** Datos con estructura fuerte: 90 % de los eventos dentro del grupo. */
export function estructuraFuerte(semilla = 11) {
  const conjunto = generar({ nUsuarios: 300, nItems: 200, grupos: 5, eventosPorUsuario: 30, fuerzaGusto: 0.9, semilla });
  const p = particionTemporal(conjunto.interacciones);
  const pasado = [...p.entrenamiento, ...p.validacion];
  const r = matrizUsuarios(conjunto.nUsuarios, conjunto.nItems, pasado);
  return { conjunto, p, r };
}
