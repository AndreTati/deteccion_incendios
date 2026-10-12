"""Interfaz web (Gradio) del sistema: detector → clasificador de recortes → criterio K-de-N → alerta.

Envuelve la misma lógica que `demo.py` (clase `Sistema` y HUD), con controles en vivo:

- Fuentes: archivo de video, cámara IP (URL rtsp/http) o webcam del navegador; además una
  pestaña de imagen suelta.
- Controles: detector, clasificador on/off y su umbral, tau, K y N. Cambiarlos reinicia el
  buffer K-de-N (una ventana armada con otro N no se puede reutilizar), sin cortar el video.
- Salidas: frame con el HUD de la demo, estado NORMAL / PRE-ALERTA / ALERTA, contador K/N
  por clase, ms/frame y registro de eventos. Cada alerta dispara además un aviso en pantalla.

Cada pestaña del navegador tiene su propio estado de alerta.

Uso:
    python app.py                 # http://127.0.0.1:7860
    python app.py --share         # link público temporal de Gradio
"""

from __future__ import annotations

import argparse
import threading
import time
from functools import lru_cache
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO

import alerta
from demo import NOMBRES, Sistema, dibujar_hud, formato_t, texto_evento

RAIZ = Path(__file__).parent
# etiqueta → (pesos, imgsz con que se entrenó)
DETECTORES = {
    "R1 · yolo11n 640 (con negativas)": ("runs/yolo11n_640_neg_s42/weights/best.pt", 640),
    "R3 · yolo11m 640 (con negativas)": ("runs/yolo11m_640_neg_s42/weights/best.pt", 640),
    "R4 · yolo11n 960 (con negativas)": ("runs/yolo11n_960_neg_s42/weights/best.pt", 960),
    "R5 · yolo11n 640 (sin negativas)": ("runs/yolo11n_640_noneg_s42/weights/best.pt", 640),
}
DETECTORES = {k: v for k, v in DETECTORES.items() if (RAIZ / v[0]).exists()}
CLASIFICADOR = "runs/clasificador/weights/best.pt"
ANCHO_MAX = 1280  # el frame anotado se achica para mandarlo al navegador
COLOR_ESTADO = {alerta.NORMAL: "#2e7d32", alerta.PRE_ALERTA: "#ef6c00", alerta.ALERTA: "#c62828"}
COLUMNAS_EVENTOS = ["t", "clase", "conf", "frames", "frame"]

# Ultralytics no es thread-safe: un frame a la vez por modelo, aunque haya varias sesiones
LOCK_MODELOS = threading.Lock()


@lru_cache(maxsize=None)
def cargar(pesos: str) -> YOLO:
    return YOLO(RAIZ / pesos)


class Sesion:
    """Estado de una pestaña del navegador: parámetros, sistema (buffer K-de-N) y eventos."""

    def __init__(self):
        self.params: tuple | None = None
        self.sistema: Sistema | None = None
        self.reiniciar()

    def reiniciar(self):
        self.sistema, self.ms_ema, self.frame, self.t_ini = None, None, 0, 0.0
        self.log: list[str] = []
        self.eventos: list[dict] = []

    def fijar_params(self, detector, usar_clf, umbral_clf, tau, k, n):
        params = (detector, bool(usar_clf), float(umbral_clf), float(tau), int(k), int(n))
        if params == self.params:
            return
        try:
            alerta.ConfigAlerta(tau=tau, k=int(k), n=int(n))
        except ValueError as e:
            gr.Warning(str(e))
            return
        self.params, self.sistema = params, None  # el sistema se reconstruye en el próximo frame

    def obtener_sistema(self) -> Sistema:
        if self.sistema is None:
            detector, usar_clf, umbral_clf, tau, k, n = self.params
            pesos, imgsz = DETECTORES[detector]
            cfg = alerta.ConfigAlerta(tau=tau, k=k, n=n, umbral_clf=umbral_clf if usar_clf else None)
            self.sistema = Sistema(cargar(pesos), cfg, cargar(CLASIFICADOR) if usar_clf else None, imgsz=imgsz)
        return self.sistema

    def paso(self, img_bgr: np.ndarray, t: float):
        """Procesa un frame y devuelve (frame anotado RGB, HTML de estado, tabla de eventos)."""
        sistema = self.obtener_sistema()
        with LOCK_MODELOS:
            res = sistema.procesar(self.frame, t, img_bgr)
        self.frame += 1
        self.ms_ema = res["ms"] if self.ms_ema is None else 0.9 * self.ms_ema + 0.1 * res["ms"]
        for ev in res["estado"]["eventos"]:
            self.eventos.append(ev)
            self.log.append(texto_evento(ev))
            gr.Warning(self.log[-1], title="ALERTA")
        hud = dibujar_hud(img_bgr, res, 1000 / self.ms_ema, self.log, sistema.cfg)
        return a_rgb(hud), html_estado(res["estado"], self.ms_ema), self.tabla()

    def tabla(self) -> list[list]:
        return [[formato_t(ev["t"]), NOMBRES.get(ev["clase"], ev["clase"]), round(ev["conf"], 2),
                 f"{ev['hits']}/{ev['n']}", ev["frame"]] for ev in self.eventos]


SESIONES: dict[str, Sesion] = {}


def sesion(request: gr.Request) -> Sesion:
    return SESIONES.setdefault(request.session_hash, Sesion())


def a_rgb(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if w > ANCHO_MAX:
        img_bgr = cv2.resize(img_bgr, (ANCHO_MAX, round(h * ANCHO_MAX / w)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def html_estado(estado: dict | None, ms: float | None = None) -> str:
    if estado is None:
        return "<div style='padding:16px;border-radius:8px;background:#616161;color:#fff;font-size:1.4em'>SIN SEÑAL</div>"
    clases = " · ".join(f"{NOMBRES.get(c, c)} {v['hits']}/{v['n']}" for c, v in estado["por_clase"].items())
    lat = f" · {ms:.0f} ms/frame" if ms is not None else ""
    return (f"<div style='padding:16px;border-radius:8px;background:{COLOR_ESTADO[estado['estado']]};color:#fff'>"
            f"<div style='font-size:1.8em;font-weight:700'>{estado['estado']}</div>"
            f"<div style='font-size:1.05em;margin-top:4px'>{clases}{lat}</div></div>")


# --- Fuentes -----------------------------------------------------------------------------------

def correr_video(video, url, vid_stride, tiempo_real, detector, usar_clf, umbral_clf, tau, k, n, request: gr.Request):
    ses = sesion(request)
    ses.reiniciar()
    ses.fijar_params(detector, usar_clf, umbral_clf, tau, k, n)
    fuente = (url or "").strip() or video
    if not fuente:
        raise gr.Error("Subí un video o escribí la URL de una cámara IP")
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        raise gr.Error(f"No se pudo abrir {fuente}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    usar_reloj = not fps or fps <= 0 or fps > 240  # cámaras IP: fps poco confiable → reloj real
    t_ini, idx = time.perf_counter(), -1
    try:
        while True:
            idx += 1
            if idx % int(vid_stride):
                if not cap.grab():
                    break
                continue
            ok, img = cap.read()
            if not ok:
                break
            t = time.perf_counter() - t_ini if usar_reloj else idx / fps
            yield ses.paso(img, t)
            if tiempo_real and not usar_reloj and (espera := t - (time.perf_counter() - t_ini)) > 0:
                time.sleep(espera)
    finally:
        cap.release()
    gr.Info(f"Fin del video: {len(ses.eventos)} alertas")


def correr_webcam(img_rgb, detector, usar_clf, umbral_clf, tau, k, n, request: gr.Request):
    if img_rgb is None:
        return gr.skip(), html_estado(None), gr.skip()
    ses = sesion(request)
    ses.fijar_params(detector, usar_clf, umbral_clf, tau, k, n)
    if ses.frame == 0:
        ses.t_ini = time.perf_counter()
    return ses.paso(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), time.perf_counter() - ses.t_ini)


def analizar_imagen(img_rgb, detector, usar_clf, umbral_clf, tau, *_):
    """Una sola imagen: K/N = 1/1, así el estado indica si la imagen sola confirma una detección."""
    if img_rgb is None:
        raise gr.Error("Subí una imagen")
    ses = Sesion()
    ses.fijar_params(detector, usar_clf, umbral_clf, tau, 1, 1)
    frame, estado, _ = ses.paso(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 0.0)
    return frame, estado


def reiniciar(request: gr.Request):
    sesion(request).reiniciar()
    return html_estado(None), []


def cambiar_params(detector, usar_clf, umbral_clf, tau, k, n, request: gr.Request):
    sesion(request).fijar_params(detector, usar_clf, umbral_clf, tau, k, n)


def cerrar_sesion(request: gr.Request):
    SESIONES.pop(request.session_hash, None)


# --- Interfaz ----------------------------------------------------------------------------------

def construir() -> gr.Blocks:
    with gr.Blocks(title="Detección temprana de incendios") as app:
        gr.Markdown("# Detección temprana de incendios\nDetector YOLO11 → clasificador de recortes → "
                    "criterio temporal K-de-N → alerta. Los controles se aplican en vivo y reinician la ventana K-de-N.")
        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                detector = gr.Dropdown(list(DETECTORES), value=next(iter(DETECTORES)), label="Detector")
                usar_clf = gr.Checkbox(True, label="Clasificador de recortes")
                umbral_clf = gr.Slider(0.05, 0.95, 0.50, step=0.05, label="Umbral del clasificador (p_real)")
                tau = gr.Slider(0.05, 0.95, 0.40, step=0.05, label="τ (confianza mínima del detector)")
                k = gr.Slider(1, 10, 3, step=1, label="K (frames con detección)")
                n = gr.Slider(1, 10, 5, step=1, label="N (ventana de frames)")
                estado = gr.HTML(html_estado(None))
                btn_reiniciar = gr.Button("Reiniciar alerta")
            controles = [detector, usar_clf, umbral_clf, tau, k, n]

            with gr.Column(scale=3):
                with gr.Tab("Video / cámara IP"):
                    with gr.Row():
                        video = gr.Video(label="Archivo de video", sources=["upload"], height=200)
                        with gr.Column():
                            url = gr.Textbox(label="o URL de cámara IP", placeholder="rtsp://usuario:clave@ip:554/stream")
                            vid_stride = gr.Slider(1, 30, 1, step=1, label="Procesar 1 de cada s frames",
                                                   info="Apuntar a ~5 fps procesados: 10 en un video de 50 fps, 6 en uno de 30")
                            tiempo_real = gr.Checkbox(True, label="Reproducir a velocidad real")
                    with gr.Row():
                        btn_iniciar = gr.Button("Iniciar", variant="primary")
                        btn_detener = gr.Button("Detener", variant="stop")
                    salida_video = gr.Image(label="Video anotado", interactive=False)
                with gr.Tab("Webcam"):
                    gr.Markdown("Se procesa un frame cada 0,2 s (~5 fps), así K/N se cuenta igual que en video.")
                    with gr.Row():
                        cam = gr.Image(sources=["webcam"], streaming=True, label="Webcam")
                        salida_cam = gr.Image(label="Webcam anotada", interactive=False)
                with gr.Tab("Imagen"):
                    gr.Markdown("Una sola imagen con K/N = 1/1: ALERTA si alguna detección queda confirmada.")
                    with gr.Row():
                        img = gr.Image(sources=["upload", "clipboard"], label="Imagen")
                        salida_img = gr.Image(label="Imagen anotada", interactive=False)
                    btn_img = gr.Button("Analizar", variant="primary")
                eventos = gr.Dataframe(headers=COLUMNAS_EVENTOS, label="Registro de alertas", interactive=False)

        # K no puede superar N
        n.change(lambda n_, k_: gr.update(maximum=n_, value=min(k_, n_)), [n, k], k)
        for c in controles:
            c.change(cambiar_params, controles, None)

        ev_video = btn_iniciar.click(correr_video, [video, url, vid_stride, tiempo_real, *controles],
                                     [salida_video, estado, eventos])
        btn_detener.click(None, cancels=[ev_video])
        cam.stream(correr_webcam, [cam, *controles], [salida_cam, estado, eventos],
                   stream_every=0.2, time_limit=None, concurrency_limit=None)
        btn_img.click(analizar_imagen, [img, *controles], [salida_img, estado])
        btn_reiniciar.click(reiniciar, None, [estado, eventos])
        app.unload(cerrar_sesion)
    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="link público temporal de Gradio")
    args = ap.parse_args()
    if not DETECTORES:
        raise SystemExit("No se encontró ningún runs/*/weights/best.pt")
    construir().queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
