import type { Interaccion } from "./datos.ts";

export interface Particion {
  entrenamiento: Interaccion[];
  validacion: Interaccion[];
  prueba: Interaccion[];
}

/**
 * Partición temporal por usuario. Para cada usuario se ordenan sus eventos
 * por tiempo; la última fracción `fraccionPrueba` va a prueba, la fracción
 * anterior `fraccionValidacion` a validación y el resto a entrenamiento.
 *
 * Por qué no aleatoria: un recomendador se usa para predecir lo que el usuario
 * hará DESPUÉS. Con una partición aleatoria, el modelo entrena con eventos
 * posteriores a los que debe predecir (por ejemplo, ve que alguien vio la
 * tercera temporada y "predice" la segunda), y las métricas salen infladas.
 * Además, las repeticiones de un mismo ítem quedarían repartidas a ambos lados
 * y el modelo "acertaría" algo que ya vio.
 *
 * Eventos repetidos: si un ítem aparece en prueba pero el usuario ya lo había
 * tocado antes (en entrenamiento o validación), se quita de prueba, porque el
 * recomendador nunca devuelve ítems ya vistos y contarlo sería penalizarlo
 * por una regla correcta. Ídem para validación respecto de entrenamiento.
 *
 * Límite conocido: la partición es por usuario, no por un corte global de
 * calendario, así que el modelo puede aprender de eventos de OTROS usuarios
 * posteriores al momento de prueba de uno dado. Con datos reales y un corte
 * global ese efecto desaparece, a cambio de dejar usuarios sin historia.
 */
export function particionTemporal(
  interacciones: readonly Interaccion[],
  fraccionPrueba = 0.2,
  fraccionValidacion = 0.2,
  minimoEntrenamiento = 3,
): Particion {
  const porUsuario = new Map<number, Interaccion[]>();
  for (const x of interacciones) {
    let l = porUsuario.get(x.usuario);
    if (!l) porUsuario.set(x.usuario, (l = []));
    l.push(x);
  }
  const p: Particion = { entrenamiento: [], validacion: [], prueba: [] };
  for (const eventos of porUsuario.values()) {
    // Orden estable por tiempo; el empate se rompe por posición original.
    const orden = eventos.map((e, k) => [e, k] as const).sort((a, b) => a[0].tiempo - b[0].tiempo || a[1] - b[1]).map((x) => x[0]);
    const n = orden.length;
    let nPrueba = Math.floor(n * fraccionPrueba);
    let nVal = Math.floor(n * fraccionValidacion);
    if (n - nPrueba - nVal < minimoEntrenamiento) {
      nPrueba = Math.min(nPrueba, Math.max(0, n - minimoEntrenamiento));
      nVal = Math.min(nVal, Math.max(0, n - minimoEntrenamiento - nPrueba));
    }
    // Un empate en el tiempo justo en el corte se manda entero al lado
    // anterior: ningún evento de prueba puede ser simultáneo a uno de
    // entrenamiento, o el "futuro" quedaría a la vista.
    let corteVal = n - nPrueba - nVal;
    let cortePrueba = n - nPrueba;
    while (cortePrueba < n && cortePrueba > 0 && orden[cortePrueba]!.tiempo === orden[cortePrueba - 1]!.tiempo) cortePrueba++;
    corteVal = Math.min(corteVal, cortePrueba);
    while (corteVal < cortePrueba && corteVal > 0 && orden[corteVal]!.tiempo === orden[corteVal - 1]!.tiempo) corteVal++;

    const ent = orden.slice(0, corteVal);
    const vistosEnt = new Set(ent.map((e) => e.item));
    const val = orden.slice(corteVal, cortePrueba).filter((e) => !vistosEnt.has(e.item));
    const vistosAntes = new Set([...vistosEnt, ...val.map((e) => e.item)]);
    const pru = orden.slice(cortePrueba).filter((e) => !vistosAntes.has(e.item));
    p.entrenamiento.push(...ent);
    p.validacion.push(...val);
    p.prueba.push(...pru);
  }
  return p;
}
