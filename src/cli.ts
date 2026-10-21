#!/usr/bin/env node
import { readFileSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { fila, matrizUsuarios, type Conjunto } from "./datos.ts";
import { arranqueEnFrio, comparar, metodos, tablaFrio, tablaMarkdown, type Hiper } from "./experimentos.ts";
import { leerMovieLens } from "./movielens.ts";
import { esExplicable, recomendar } from "./modelos/modelo.ts";
import { cargar, guardar, type Guardado } from "./persistencia.ts";
import { generar } from "./sintetico.ts";

const AYUDA = `uso:
  recomendador entrenar  --datos <fuente> --metodo <m> [--hiper clave=valor,...] --salida modelo.json
  recomendador recomendar --modelo modelo.json --usuario <id> [--k 10] [--explicar]
  recomendador evaluar   --datos <fuente> [--k 10] [--rapido] [--frio]

fuente: "sintetico" (por omisión, semilla 42), "sintetico:<semilla>" o "ml100k:<carpeta ml-100k>"
métodos: popularidad | segmento | knn-coseno | knn-ajustada | als | bpr`;

const ALIAS: Record<string, string> = {
  popularidad: "popularidad",
  segmento: "popularidad por segmento",
  "knn-coseno": "item-kNN coseno",
  "knn-ajustada": "item-kNN ajustada",
  als: "ALS implícito",
  bpr: "BPR",
};

function cargarDatos(fuente: string): Conjunto {
  if (fuente === "sintetico") return generar();
  if (fuente.startsWith("sintetico:")) return generar({ semilla: Number(fuente.slice(10)) });
  if (fuente.startsWith("ml100k:")) return leerMovieLens(fuente.slice(7));
  throw new Error(`fuente desconocida: ${fuente}`);
}

function leerHiper(texto: string | undefined): Hiper | undefined {
  if (!texto) return undefined;
  const h: Hiper = {};
  for (const par of texto.split(",")) {
    const [k, v] = par.split("=");
    if (!k || v === undefined) throw new Error(`hiperparámetro mal escrito: ${par}`);
    h[k] = Number.isNaN(Number(v)) ? v : Number(v);
  }
  return h;
}

function main(argv: string[]): void {
  const [comando, ...resto] = argv;
  const { values: o } = parseArgs({
    args: resto,
    options: {
      datos: { type: "string", default: "sintetico" },
      metodo: { type: "string", default: "als" },
      hiper: { type: "string" },
      salida: { type: "string" },
      modelo: { type: "string" },
      usuario: { type: "string" },
      k: { type: "string", default: "10" },
      explicar: { type: "boolean", default: false },
      rapido: { type: "boolean", default: false },
      frio: { type: "boolean", default: false },
    },
  });
  const k = Number(o.k);

  if (comando === "entrenar") {
    const conjunto = cargarDatos(o.datos);
    const nombre = ALIAS[o.metodo];
    const metodo = metodos(conjunto).find((m) => m.nombre === nombre);
    if (!metodo) throw new Error(`método desconocido o sin datos para él: ${o.metodo}`);
    const h = leerHiper(o.hiper) ?? metodo.grilla[0]!;
    const r = matrizUsuarios(conjunto.nUsuarios, conjunto.nItems, conjunto.interacciones);
    const t0 = performance.now();
    const modelo = metodo.entrenar(r, h, conjunto);
    const salida = o.salida ?? "modelo.json";
    writeFileSync(salida, JSON.stringify(guardar(modelo, r, conjunto.nombresItems)));
    console.log(`${modelo.nombre} entrenado con ${conjunto.interacciones.length} interacciones en ${((performance.now() - t0) / 1000).toFixed(2)} s -> ${salida}`);
    return;
  }

  if (comando === "recomendar") {
    if (!o.modelo || o.usuario === undefined) throw new Error("faltan --modelo y --usuario");
    const g = JSON.parse(readFileSync(o.modelo, "utf8")) as Guardado;
    const { modelo, historia } = cargar(g);
    const u = Number(o.usuario);
    if (!Number.isInteger(u) || u < 0 || u >= g.nUsuarios) throw new Error(`usuario fuera de rango (0..${g.nUsuarios - 1})`);
    const nombre = (i: number) => g.nombresItems?.[i] ?? `item-${i}`;
    const vistos = fila(historia, u).indices;
    console.log(`usuario ${u}: ${vistos.length} ítems en su historia; modelo ${modelo.nombre}`);
    const puntajes = new Float64Array(g.nItems);
    modelo.puntuar(u, puntajes);
    recomendar(modelo, u, k, vistos, g.nItems).forEach((i, pos) => {
      console.log(`${String(pos + 1).padStart(2)}. ${nombre(i)}  (puntaje ${puntajes[i]!.toFixed(3)})`);
      if (o.explicar && esExplicable(modelo)) {
        const ex = modelo.explicar(u, i, 3);
        if (ex.length) console.log(`    porque viste: ${ex.map((a) => `${nombre(a.item)} [${a.aporte.toFixed(3)}]`).join("; ")}`);
      }
    });
    return;
  }

  if (comando === "evaluar") {
    const conjunto = cargarDatos(o.datos);
    console.log(`${conjunto.nUsuarios} usuarios, ${conjunto.nItems} ítems, ${conjunto.interacciones.length} interacciones`);
    const filas = comparar(conjunto, k, o.rapido, (m) => process.stderr.write(`entrenando ${m}...\n`));
    console.log(tablaMarkdown(filas, k));
    if (o.frio) {
      console.log("\narranque en frío (10 % de ítems sin historia):\n");
      console.log(tablaFrio(arranqueEnFrio(conjunto, k), k));
    }
    return;
  }

  console.log(AYUDA);
  if (comando !== undefined && comando !== "ayuda") process.exitCode = 1;
}

try {
  main(process.argv.slice(2));
} catch (e) {
  console.error(`error: ${(e as Error).message}`);
  process.exitCode = 1;
}
