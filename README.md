# Detección temprana de incendios: fuego y humo sobre D-Fire

Trabajo práctico final de Visión por Computadora II (CEIA). El proyecto plantea un sistema de alerta temprana de incendios (fuego y humo) sobre el dataset [D-Fire](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo), evaluado a nivel frame (detección por clase, falsos positivos en imágenes negativas y latencia) y sobre video, con un clasificador de recortes y un criterio temporal K-de-N que exige detecciones consistentes antes de disparar una alerta.

## Entorno

El entorno se gestiona con conda y está definido en [`environment.yml`](environment.yml).

Crear el entorno:

```bash
conda env create -f environment.yml
```

Activarlo:

```bash
conda activate incendios-tp
```

Si se modifica `environment.yml` después de creado el entorno:

```bash
conda env update -f environment.yml --prune
```

## Estructura

| Archivo | Qué hace |
|---|---|
| `descarga.ipynb` | Descarga D-Fire, verifica splits, orden de clases e integridad de labels; genera `configs/` |
| `eda.ipynb` | Análisis exploratorio: composición, tamaños de caja, resoluciones, duplicados entre splits |
| `train.ipynb` | Entrena y evalúa un detector por ejecución (Kaggle / Colab / local) → `results/master.csv` |
| `clasificador.ipynb` | Dataset de recortes, entrenamiento y evaluación del clasificador liviano (H4) |
| `benchmark.ipynb` | Latencia del detector y del sistema completo, CPU y GPU |
| `inferencia_video.ipynb` | Video → CSV de detecciones por frame (+ `p_real` del clasificador) |
| `barrido_alerta.ipynb` | Barrido de τ / K / N con y sin clasificador → falsas alarmas/hora vs. latencia (H3, H4) |
| `latencia_figlib.py` | H3 sobre secuencias de cámara fija de FIgLib (una foto por minuto, tiempo real): latencia de alerta y falsas alarmas para τ × K/N, con y sin clasificador. Clips en `configs/figlib_clips.json` |
| `analisis_errores.ipynb` | Peores casos por clase, recall por tamaño (H1), con vs. sin negativas (H2), set de estrés |
| `alerta.py` | Criterio K-de-N y estados NORMAL / PRE-ALERTA / ALERTA |
| `recortes.py` | Recorte con margen para el clasificador (idéntico al entrenar y al usar) |
| `evaluacion.py` | Emparejamiento predicción ↔ ground truth a un punto de operación |
| `demo.py` | Video → sistema completo → MP4 anotado con HUD |
| `app.py` | Interfaz web (Gradio): archivo de video, cámara IP o webcam, con τ, K, N y clasificador ajustables en vivo |

`train`, `clasificador`, `benchmark` y `analisis_errores` tienen `SMOKE_TEST = True` para probar el pipeline sin GPU ni modelos entrenados; `inferencia_video` usa pesos de COCO si todavía no hay detector.

## Cómo reproducir

1. `descarga.ipynb` y `eda.ipynb` (local).
2. **Detectores** con `train.ipynb` (GPU: Kaggle o Colab), cambiando `RUN_ID` en cada ejecución (`BEST_SIZE = "n"`, ver abajo). R5 tiene que estar antes que el clasificador, que se entrena con sus falsos positivos.
3. **Clasificador** con `clasificador.ipynb` (`MINE_RUN` = R5, `EVAL_RUN` = R1). Fuera de local, subir `recortes.py` y `evaluacion.py` a la sesión.

   **Traer resultados de Colab / Kaggle** (detectores y clasificador): bajar `<run>_resultados.zip` de `Drive/incendios-tp/`, descomprimirlo en la raíz del repo (`Expand-Archive -Path <zip> -DestinationPath .`) y correr la última celda de `train.ipynb` (`merge_master`), que agrega la fila del run a `results/master.csv` sin tocar las demás.
4. **Videos:** copiarlos a `videos/`, completar `videos/clips.csv` y anotar `t_evento_s` (criterio en `inferencia_video.ipynb`).
5. `inferencia_video.ipynb` → `barrido_alerta.ipynb`.
6. **Set de estrés:** imágenes en `data/estres/<categoria>/`. Después `analisis_errores.ipynb` (completar a mano `results/errores_causas.csv` y volver a correr la celda de resumen).
7. `benchmark.ipynb` en local (CPU; es la medición de despliegue) y, opcionalmente, en Colab (GPU).
8. **Latencia de alerta en cámaras fijas (H3):** bajar los MP4 de las secuencias de FIgLib de `configs/figlib_clips.json` ([Meadowfire](https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/20190716_Meadowfire_hp-n-mobo-c/index.html), [Lyons Peak](https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/20190717_FIRE_lp-n-mobo-c/index.html)) a la ruta que indica `video` y correr, en CPU:

   ```bash
   python latencia_figlib.py                                                   # R1 a 640
   python latencia_figlib.py --detector runs/yolo11n_960_neg_s42/weights/best.pt --imgsz 960
   python latencia_figlib.py --detector runs/yolo11m_640_neg_s42/weights/best.pt
   ```

   Cada frame del MP4 es una foto (~1 min); el script usa el tiempo real de cada foto, así que K/N se cuenta en minutos.
9. Demo:

   ```bash
   python demo.py --video videos/humo.mp4 --salida demo_humo.mp4 \
       --detector runs/yolo11n_640_neg_s42/weights/best.pt \
       --clasificador runs/clasificador/weights/best.pt --tau 0.4 --k 3 --n 5 --vid-stride 10
   ```

   `--n` se cuenta en frames procesados: elegir `--vid-stride` para procesar unos 5 fps (10 en un video de 50 fps, 6 en uno de 30), así la ventana de 5 frames cubre 1 s.
10. Interfaz web:

    ```bash
    python app.py            # http://127.0.0.1:7860 (--share para un link público temporal)
    ```

    Pestañas de video (archivo o URL rtsp/http de una cámara IP), webcam (un frame cada 0,2 s) e imagen suelta. El panel lateral elige el detector y ajusta τ, K, N y el clasificador mientras corre el video; cada cambio reinicia la ventana K-de-N. Muestra el mismo HUD que `demo.py`, el estado NORMAL / PRE-ALERTA / ALERTA, el registro de alertas y un aviso en pantalla en cada alerta.

## Runs

| Run | Modelo | imgsz | Train | Responde | Estado |
|---|---|---|---|---|---|
| R1 | `yolo11n` | 640 | con negativas | baseline, Pareto capacidad/latencia | ✅ |
| R2 | `yolo11s` | 640 | con negativas | Pareto capacidad/latencia | no se corrió |
| R3 | `yolo11m` | 640 | con negativas | Pareto capacidad/latencia | ✅ |
| R4 | `yolo11n` | 960 | con negativas | H1: resolución | ✅ |
| R5 | `yolo11n` | 640 | sin negativas | H2: negativas | ✅ |
| CLF | `yolo11n-cls` | 224 | recortes (FP de R5) | H4: clasificador de recortes | ✅ |

Nombre de cada run: `{modelo}_{imgsz}_{neg|noneg}_s{seed}`, seed 42, 50 épocas, batch 16.

**Tamaño elegido: `n`.** R3 (`m`) mejora +0,020 de mAP50 en test (0,759 → 0,779) pero es ~5× más lento en CPU (56 → 299 ms/frame en un i7-1355U, `results/benchmark.csv`), lo que también alarga la latencia de alerta. Por eso R4 y R5 usan `n`.

Resultados de test (mAP50 / mAP50-95): R1 0,759 / 0,434 · R3 0,779 / 0,455 · R4 0,773 / 0,444 · R5 0,699 / 0,402. Todas las métricas en `results/master.csv`.

Clasificador de recortes (sobre R1, τ = 0,40, `p_real ≥ 0,5`): en test, las imágenes negativas con algún falso positivo bajan de 0,65 % a 0,10 %, a costa de −4,5 puntos de recall en humo y −0,9 en fuego (`figures/clasificador_recall_vs_fp.png`). Como R1 ya casi no se equivoca con las negativas del test (son casi duplicados de train), su efecto real se mide sobre escenas externas.

Prueba de estrés (`results/estres_video.csv`): cuatro timelapses sin incendio (niebla, atardecer y noche, dos nublados; 156 s, 780 frames a 5 fps) con R1, τ = 0,40 y K/N = 3/5. El detector marca humo en 7 frames y se emiten 2 falsas alarmas; con el clasificador quedan 3 frames y 1 falsa alarma (descarta una tormenta, no la niebla baja de un valle).

Latencia de alerta en cámaras fijas (`results/alerta_figlib_*.csv`): dos secuencias de FIgLib, una foto por minuto desde 40 min antes de la ignición. En Meadowfire, R1 (τ = 0,40) no da falsas alarmas en los 49 min previos y alerta a los 4, 5 y 6 min de que el humo es visible con K/N = 1/1, 2/2 y 3/5: el criterio temporal suma K − 1 fotos. En Lyons Peak ningún detector supera τ = 0,25 sobre el humo (máximo 0,19–0,24) y no hay alerta. R4 (960 px) detecta el humo 2 min más tarde y marca como humo una mancha de rocas en las 70 fotos (alerta permanente). Con R3, τ = 0,15, clasificador y K/N ≥ 2/2 se alerta en ambas secuencias sin falsas alarmas previas (4–5 y 5–6 min), con el umbral elegido sobre esas mismas secuencias.

No se corrieron el barrido de `barrido_alerta.ipynb` (falsas alarmas por hora sobre video de vigilancia en tiempo real) ni el set de estrés por imágenes de `analisis_errores.ipynb`.
