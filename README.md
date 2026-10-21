# recomendador

Sistema de recomendación sobre interacciones implícitas (clics, compras,
reproducciones), escrito en TypeScript sin dependencias de ejecución. Incluye
los modelos de base que cualquier propuesta tiene que superar, dos
factorizaciones de matrices, arranque en frío por atributos y un protocolo de
evaluación que no se hace trampas.

## El problema

Con datos implícitos solo se sabe lo que el usuario tocó; lo que no tocó puede
ser desinterés o desconocimiento. La pregunta práctica es qué mostrarle
después, y la difícil es cómo saber si un modelo lo hace mejor que "lo más
vendido". Para eso hacen falta tres cosas: una partición que no deje ver el
futuro, métricas de ranking además de precisión, y datos donde se conozca la
respuesta correcta.

## Diseño

```
src/
  datos.ts         interacciones y matriz dispersa CSR
  sintetico.ts     generador con grupos de gusto y popularidad de Zipf
  movielens.ts     lector de MovieLens 100K (no incluido)
  particion.ts     partición temporal por usuario
  algebra.ts       Cholesky y Gram, a mano
  modelos/
    popularidad.ts popularidad global y por segmento
    knn.ts         item-kNN, coseno y coseno ajustado
    als.ts         ALS implícito (Hu, Koren, Volinsky 2008)
    bpr.ts         BPR-MF con SGD (Rendle et al. 2009)
    frio.ts        contenido e híbrido para ítems nuevos
  metricas.ts      P@k, R@k, NDCG@k, AP@k
  evaluar.ts       evaluación, cobertura, novedad, sesgo de popularidad
  experimentos.ts  búsqueda de hiperparámetros y arranque en frío
  persistencia.ts  guardar y cargar modelos en JSON
  cli.ts
```

### Modelos

- **Popularidad global**: el piso.
- **Popularidad por segmento**: lo popular dentro del segmento observable del
  usuario (en los datos sintéticos, un segmento que coincide con su grupo real
  la mitad de las veces; en MovieLens, la ocupación). Se encoge hacia la
  popularidad global con un peso `m` para que un segmento chico no dependa de
  tres personas.
- **Item-kNN**: similitud coseno entre columnas de la matriz, con peso
  `log(1 + valor)` y encogimiento por usuarios en común. La variante
  "ajustada" resta el promedio de cada usuario antes de comparar (Sarwar et
  al. 2001); solo aporta cuando los valores varían, porque con datos binarios
  todo queda en cero.
- **ALS implícito**: confianza `c = 1 + α·r`, preferencia binaria, pérdida
  sobre todas las celdas (también las vacías). Cada medio paso resuelve un
  sistema `f x f` por usuario o por ítem con el truco del artículo
  (`YᵀY` una vez, más la corrección por lo observado) y Cholesky propio. Cada
  medio paso minimiza exactamente su bloque, así que la pérdida no puede
  subir; hay una prueba que lo verifica en cada paso.
- **BPR**: aprende a ordenar un ítem visto por encima de uno no visto, con SGD
  sobre tríos muestreados y sesgo por ítem.
- **Contenido**: perfil del usuario como promedio de los atributos de lo que
  tocó, coseno contra cada ítem.
- **Híbrido**: ALS donde hay historia; para un ítem sin interacciones, una
  regresión ridge de atributos a factores, aprendida sobre los ítems con
  historia, le da un vector en el mismo espacio.

### Explicaciones

Item-kNN y ALS responden "porque viste X". En kNN el puntaje es una suma de
similitudes con lo visto, y cada término es el aporte. En ALS se usa la
descomposición de la sección 5 de Hu et al.: con `W_u = (YᵀC_uY + λI)⁻¹`, el
puntaje es `Σ_j c_uj · y_iᵀ W_u y_j` sobre la historia. Para que la suma sea
exacta, el entrenamiento termina con un paso de usuarios extra; una prueba
comprueba que los aportes suman el puntaje con ocho decimales.

### Datos sintéticos

Cada usuario y cada ítem pertenecen a uno de `k` grupos de gusto. Un evento
cae en el grupo del usuario con probabilidad `fuerzaGusto` (0,8 por omisión)
y si no en cualquier parte; dentro de eso, los ítems pesan según Zipf, así
hay éxitos y cola larga. Los atributos de los ítems son la indicadora del
grupo más ruido, más dos dimensiones de ruido puro. Como la verdad se conoce,
las pruebas pueden preguntar si un modelo la recupera.

### Evaluación

- **Partición temporal por usuario**: los eventos de cada usuario se ordenan
  por tiempo; el último 20 % va a prueba, el 20 % anterior a validación y el
  resto a entrenamiento. Una partición aleatoria deja que el modelo aprenda de
  eventos posteriores a los que debe predecir y infla las métricas. Un empate
  de tiempo en el corte se manda entero al lado posterior, y un ítem que
  reaparece en prueba después de haberlo tocado antes se quita, porque el
  recomendador nunca devuelve lo visto.
- **Métricas**: Precision@k, Recall@k, NDCG@k, MAP@k (AP normalizado por
  `min(|relevantes|, k)`), cobertura del catálogo, novedad (autoinformación
  media `−log2` de la fracción de usuarios que tocó lo recomendado) y
  popularidad media de lo recomendado como medida de sesgo.
- **Hiperparámetros**: cada punto de la grilla se entrena con entrenamiento y
  se mide en validación (NDCG@10). Con el mejor se reentrena con
  entrenamiento + validación y se mide una sola vez en prueba.

## Uso

Requiere Node 24.

```
npm install
npm run build
node dist/cli.js evaluar --frio            # grilla completa, unos 35 s
node dist/cli.js evaluar --rapido          # un punto por método, menos de 1 s
node dist/cli.js entrenar --metodo als --hiper factores=32,lambda=10,alfa=1 --salida als.json
node dist/cli.js recomendar --modelo als.json --usuario 3 --k 5 --explicar
```

Métodos: `popularidad`, `segmento`, `knn-coseno`, `knn-ajustada`, `als`,
`bpr`. Fuentes de datos: `sintetico` (semilla 42), `sintetico:<semilla>` o
`ml100k:<carpeta>`.

### MovieLens 100K

No viene en el repositorio porque su licencia pide bajarlo del sitio de
GroupLens (https://grouplens.org/datasets/movielens/100k/). Se descomprime y
se pasa la carpeta:

```
unzip ml-100k.zip
node dist/cli.js evaluar --datos ml100k:./ml-100k --frio
```

El lector toma `u.data` (cada nota es una interacción y la nota queda como
valor, así la confianza de ALS crece con ella), los 19 géneros de `u.item`
como atributos y la ocupación de `u.user` como segmento. `leerMovieLens`
acepta una nota mínima para quedarse solo con lo que gustó.

## Resultados

Salida real de `node dist/cli.js evaluar --frio` sobre los datos sintéticos
por omisión, en un Mac. Prueba = último 20 % de cada usuario, k = 10.

```
| método | P@10 | R@10 | NDCG@10 | MAP@10 | cobertura | novedad (bits) | pop. media | NDCG@10 val. | hiperparámetros | s |
|---|---|---|---|---|---|---|---|---|---|---|
| popularidad | 0.044 | 0.160 | 0.125 | 0.075 | 0.052 | 2.05 | 0.255 | 0.129 | - | 0.0 |
| popularidad por segmento | 0.069 | 0.245 | 0.175 | 0.104 | 0.302 | 2.71 | 0.195 | 0.174 | m=1 | 0.2 |
| item-kNN coseno | 0.104 | 0.366 | 0.274 | 0.176 | 0.435 | 3.77 | 0.095 | 0.288 | vecinos=20, encoger=10 | 0.5 |
| item-kNN ajustada | 0.072 | 0.253 | 0.182 | 0.108 | 0.725 | 3.84 | 0.099 | 0.188 | vecinos=200, encoger=10 | 0.9 |
| ALS implícito | 0.104 | 0.361 | 0.275 | 0.177 | 0.555 | 3.84 | 0.093 | 0.295 | factores=32, lambda=10, alfa=1 | 29.8 |
| BPR | 0.109 | 0.387 | 0.275 | 0.173 | 0.530 | 3.53 | 0.129 | 0.283 | factores=16, regularizacion=0.05 | 2.1 |
```

Lo que se lee ahí:

- Los tres modelos personalizados más fuertes (item-kNN coseno, ALS, BPR)
  quedan juntos, con NDCG@10 cerca de 0,275 contra 0,125 de popularidad.
  Popularidad por segmento recupera parte de la distancia con un segmento que
  solo acierta la mitad de las veces.
- Popularidad recomienda el 5 % del catálogo y con la mayor popularidad
  media; ALS y kNN cubren entre 43 % y 56 % con ítems bastante menos
  populares, que es la cola larga que el generador puso ahí.
- El coseno ajustado rinde menos: estos datos son casi binarios (pocos ítems
  repetidos) y restar el promedio borra casi toda la señal.
- ALS es lo más lento por la grilla de 18 puntos con `f` hasta 32.

Arranque en frío: se declara nuevo un 10 % de los ítems, se borran todas sus
interacciones del pasado y se ordenan solo esos ítems para cada usuario.

```
| método | P@10 | R@10 | NDCG@10 | cobertura de los nuevos |
|---|---|---|---|---|
| al azar | 0.027 | 0.226 | 0.103 | 1.000 |
| popularidad | 0.019 | 0.167 | 0.077 | 0.250 |
| ALS implícito | 0.019 | 0.167 | 0.077 | 0.250 |
| contenido | 0.044 | 0.378 | 0.212 | 1.000 |
| híbrido (ALS + atributos) | 0.052 | 0.454 | 0.241 | 1.000 |
```

Popularidad y ALS puro empatan porque los dos ven puntaje cero en todos los
ítems nuevos y el orden queda por índice; por eso rinden peor que al azar. El
contenido y el híbrido usan los atributos y duplican el NDCG del azar; el
híbrido gana porque proyecta al espacio de ALS, donde ya están los gustos del
usuario.

Recomendación explicada con ALS (usuario 3, grupo 5 según la verdad del
generador):

```
usuario 3: 25 ítems en su historia; modelo ALS implícito
 1. item-13 (grupo 5)  (puntaje 0.719)
    porque viste: item-5 (grupo 5) [0.164]; item-21 (grupo 5) [0.121]; item-341 (grupo 5) [0.054]
 2. item-293 (grupo 5)  (puntaje 0.698)
    porque viste: item-21 (grupo 5) [0.156]; item-5 (grupo 5) [0.098]; item-117 (grupo 5) [0.050]
 3. item-269 (grupo 5)  (puntaje 0.652)
    porque viste: item-5 (grupo 5) [0.158]; item-21 (grupo 5) [0.134]; item-341 (grupo 5) [0.055]
```

## Pruebas

`npm test` corre 20 pruebas de conducta, entre ellas: la pérdida de ALS no
sube en ningún medio paso; Cholesky contra una factorización conocida y
contra sistemas aleatorios; ALS y BPR superan por al menos 1,3 veces a
popularidad en NDCG cuando la estructura es fuerte; item-kNN recomienda más
de 85 % de ítems del grupo real del usuario; ningún modelo devuelve un ítem
visto; la partición no deja ningún evento de entrenamiento posterior a uno de
validación o prueba del mismo usuario; el híbrido supera a ALS puro en ítems
nuevos; un modelo guardado y cargado recomienda lo mismo.

Cada archivo de prueba anota al inicio una mutación puesta a mano y cuántas
pruebas cayeron. Una de ellas no cayó: invertir solo el gradiente del ítem
negativo en BPR degrada el modelo, pero no lo suficiente para bajarlo de 1,3
veces popularidad. Está anotado como punto ciego.

## Limitaciones

- Los números de arriba son sobre datos sintéticos que el propio proyecto
  genera; no hay resultados de MovieLens en este README porque no se bajó el
  conjunto. Con datos reales el orden entre métodos puede cambiar.
- La partición es temporal por usuario, no por un corte de calendario global:
  el modelo puede aprender de eventos de otros usuarios posteriores al momento
  de prueba de uno dado.
- Item-kNN arma una matriz densa de ítems x ítems: sirve hasta unos pocos
  miles de ítems. ALS construye cada sistema en `O(n_u f²)` sin paralelismo.
- La grilla de hiperparámetros es chica y la selección usa una sola
  partición; no hay intervalos de confianza ni varias semillas.
- BPR muestrea negativos uniformes; no hay muestreo adaptativo ni WARP.

## Referencias

- Hu, Koren, Volinsky. Collaborative Filtering for Implicit Feedback Datasets. ICDM 2008.
- Rendle, Freudenthaler, Gantner, Schmidt-Thieme. BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI 2009.
- Sarwar, Karypis, Konstan, Riedl. Item-based Collaborative Filtering Recommendation Algorithms. WWW 2001.

## Licencia

MIT.
