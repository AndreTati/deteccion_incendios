# Detección temprana de incendios — fuego y humo sobre D-Fire

Trabajo práctico final de Visión por Computadora II (CEIA). El proyecto plantea un sistema de alerta temprana de incendios (fuego y humo) sobre el dataset [D-Fire](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo), evaluado tanto a nivel frame (detección) como a nivel alerta (falsas alarmas por hora, latencia de alerta), usando un criterio temporal de frames consistentes antes de disparar una alerta.

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
