import json
import os
import queue
import re
import threading
import tkinter as tk
from tkinter import ttk, HORIZONTAL

import cv2
import numpy as np
import torch
import onnxruntime as ort
from PIL import Image, ImageTk, ImageDraw
from tkinterdnd2 import TkinterDnD, DND_FILES
from moviepy import VideoFileClip
from ultralytics import YOLO

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from tkinter import ttk, HORIZONTAL, filedialog
import imageio_ffmpeg          # già installato come dipendenza di moviepy

try:
    from insightface.app import FaceAnalysis
except ImportError:
    raise SystemExit(
        "InsightFace non è installato. Installa (meglio in un venv separato):\n"
        "  python -m pip install insightface\n"
        "  python -m pip uninstall -y onnxruntime\n"
        "  python -m pip install onnxruntime-gpu\n"
        "Se la compilazione fallisce su Windows servono i Microsoft C++ Build Tools."
    )

# ---------------------------------------------------------------- impostazioni
dim = 400                      # dimensione delle canvas
MODEL_PATH = "yolov12l-face.pt"
IMGSZ = 960                    # risoluzione di input per YOLO
CONF = 0.4                     # soglia di confidenza YOLO
pixels = 15                    # margine aggiunto su ogni lato del rettangolo
line_width = 8                 # spessore (il frame viene poi ridotto a 400 px)

# --- riconoscimento
SOGLIA_RICONOSCIMENTO = 0.40   # similarità coseno minima per dire "è la stessa persona"
                               # (più alta = più severo, più persone nuove; più bassa = più confusioni)
SOGLIA_RIMOZIONE = 0.50        # in correzione: i campioni della persona sbagliata più simili di così
                               # al volto corretto vengono tolti
AUTO_IMPARA_MIN = 0.50         # un riconoscimento sicuro ma non identico aggiunge un campione
AUTO_IMPARA_MAX = 0.75         # (più angolazioni della stessa persona = riconoscimento migliore)
MAX_CAMPIONI = 30              # campioni massimi per persona
MIN_VOLTO_PX = 40              # volti più piccoli di così (lato minore) non vengono riconosciuti
MIN_DET_SCORE = 0.5            # confidenza minima del rilevatore di InsightFace sul ritaglio
MARGINE_CROP = 0.4             # margine attorno al box YOLO per il ritaglio del volto
EMB_SIZE = 256                 # il ritaglio viene portato a questo lato massimo

# colore in base all'identità: persona 1 verde, persona 2 azzurro, persona 3 arancione, ...
COLORI = [
    (0, 200, 0),       # persona 1: verde
    (0, 191, 255),     # persona 2: azzurro
    (255, 165, 0),     # persona 3: arancione
    (255, 0, 255),     # persona 4: magenta
    (255, 230, 0),     # persona 5: giallo
    (255, 60, 60),     # persona 6: rosso
    (150, 100, 255),   # persona 7: viola
    (0, 255, 200),     # persona 8: turchese
]
COLORE_SCONOSCIUTO = (160, 160, 160)   # volto troppo piccolo/sfocato per essere riconosciuto

# --- capelli (SAM3)
# --- riconoscimento capelli (SAM3) ---

HAIR_PROMPT = (
    "human hair, female hair, woman's hairstyle, "
    "hair on the head, scalp hair, "
    "front view of woman's hair, "
    "side view of woman's hair, "
    "rear view of woman's hair, "
    "long hair, short hair, medium length hair, "
    "loose hair, straight hair, wavy hair, curly hair, "
    "hair covering forehead, "
    "hair around the face, "
    "hair covering ears, "
    "hair behind the head, "
    "hair falling over shoulders, "
    "individual hair strands, "
    "natural hair"
)

# Soglia più permissiva
HAIR_SCORE = 0.20

# Colore maschera
HAIR_COLOR = (255, 0, 128)

# Espansione della maschera
HAIR_DILATE_PX = 2

SAM3_BPE_PATH = r"G:\sam3\assets\bpe_simple_vocab_16e6.txt.gz"

# ---------------------------------------------------------------- stato globale
path_video = None
video = None
n_frames = 0
photo_input = None
photo_out = None

video_lock = threading.Lock()
request_event = threading.Event()
requested_index = 0
running = True
results = queue.Queue()

Raccoglitore_elementi_riconoscimento_facciale = []   # una voce per ogni riga "persona"
identita_ora = []                                    # identità dei volti nel frame mostrato

# ultimo frame mostrato: serve per ridisegnare i box subito dopo una correzione
ultimo_frame = None
ultimo_coords = []
ultimo_embs = []

capelli_attivi = False       # casella 'Capelli' (letta dal worker)
nomi_identita = {}             # nomi personalizzati: {identità: testo}

# ---------------------------------------------------------------- modelli
use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"
print("Dispositivo YOLO:", device)
model_face = YOLO(MODEL_PATH)
model_face.to(device)

predict_kwargs = dict(imgsz=IMGSZ, conf=CONF, device=device, verbose=False)
if use_cuda:
    predict_kwargs["quantize"] = 16    # FP16 solo su GPU CUDA

# onnxruntime-gpu: carica le DLL CUDA 12 / cuDNN installate via pip nel venv
# (nvidia-cuda-runtime-cu12, nvidia-cublas-cu12, nvidia-cufft-cu12, nvidia-cudnn-cu12).
# Richiede onnxruntime >= 1.21. Se qualcosa manca, onnxruntime ripiega da solo sulla CPU.
try:
    ort.preload_dlls()
except AttributeError:
    print("onnxruntime troppo vecchio per preload_dlls(): aggiorna con "
          "'python -m pip install -U onnxruntime-gpu'")
except Exception as e:
    print("preload_dlls non riuscito:", e)
print("Provider onnxruntime disponibili:", ort.get_available_providers())

# InsightFace: solo rilevamento (sul ritaglio) e riconoscimento (embedding ArcFace)
_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
face_app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                        providers=_providers)
face_app.prepare(ctx_id=0 if use_cuda else -1, det_size=(EMB_SIZE, EMB_SIZE))

# SAM3: segmentazione dei capelli da prompt testuale (caricato una sola volta)
if not os.path.exists(SAM3_BPE_PATH):
    raise SystemExit(f"File tokenizer SAM3 non trovato: {SAM3_BPE_PATH}\n"
                     "Imposta SAM3_BPE_PATH sul file assets\\bpe_simple_vocab_16e6.txt.gz del repo sam3.")
sam_model = build_sam3_image_model(bpe_path=SAM3_BPE_PATH, device=device)
sam_processor = Sam3Processor(sam_model)


# ---- GALLERIA INIZIO ------------------------------------------------------------------
# galleria[identità] = lista di embedding (vettori normalizzati) di quella persona
galleria = {}
galleria_lock = threading.Lock()


def prossima_identita():
    """Primo numero di identità libero (0 = persona 1, ...). Da chiamare con il lock preso."""
    k = 0
    while k in galleria:
        k += 1
    return k


def aggiungi_campione(ident, emb):
    """Aggiunge un campione; oltre MAX_CAMPIONI toglie il più simile al nuovo (tiene la varietà).
    Da chiamare con il lock preso."""
    campioni = galleria.setdefault(ident, [])
    campioni.append(emb)
    if len(campioni) > MAX_CAMPIONI:
        sim = np.stack(campioni[:-1]) @ emb
        del campioni[int(np.argmax(sim))]


def assegna_identita(embs, impara=False):
    """Associa a ogni volto (embedding o None) l'identità della galleria.
    - ogni identità può essere usata una sola volta per frame (due volti nello stesso frame
      non sono la stessa persona), assegnando prima le coppie più simili
    - volto non riconosciuto -> nuova identità (aggiunta alla galleria)
    - volto senza embedding -> -1
    - impara=True: un riconoscimento 'abbastanza sicuro ma non identico' aggiunge un campione"""
    identita = [None] * len(embs)
    with galleria_lock:
        coppie = []
        for i, e in enumerate(embs):
            if e is None:
                continue
            for ident, campioni in galleria.items():
                s = float(np.max(np.stack(campioni) @ e))
                if s >= SOGLIA_RICONOSCIMENTO:
                    coppie.append((s, i, ident))
        coppie.sort(reverse=True)

        for s, i, ident in coppie:
            if identita[i] is None and ident not in identita:
                identita[i] = ident
                if impara and AUTO_IMPARA_MIN <= s < AUTO_IMPARA_MAX:
                    aggiungi_campione(ident, embs[i])

        for i, e in enumerate(embs):
            if identita[i] is None:
                if e is None:
                    identita[i] = -1
                else:
                    ident = prossima_identita()
                    galleria[ident] = [e]
                    identita[i] = ident
    return identita


def riassegna_volti(correzioni):
    """correzioni: lista di (embedding, identità_vecchia, identità_nuova).
    Toglie dalla persona sbagliata i campioni simili al volto e lo aggiunge a quella giusta."""
    with galleria_lock:
        for emb, vecchia, nuova in correzioni:
            campioni = galleria.get(vecchia)
            if campioni:
                galleria[vecchia] = [c for c in campioni if float(c @ emb) < SOGLIA_RIMOZIONE]
                if not galleria[vecchia]:
                    del galleria[vecchia]
        for emb, vecchia, nuova in correzioni:
            aggiungi_campione(nuova, emb)
# ---- GALLERIA FINE --------------------------------------------------------------------


def percorso_persone():
    return None if path_video is None else path_video + ".persone.json"


def salva_persone():
    """Salva galleria e nomi accanto al video: alla prossima apertura le persone sono già note."""
    p = percorso_persone()
    if p is None:
        return
    with galleria_lock:
        dati = {
            "galleria": {str(k): [c.tolist() for c in v] for k, v in galleria.items()},
            "nomi": {str(k): v for k, v in nomi_identita.items()},
        }
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(dati, f)
    except OSError as e:
        print("Impossibile salvare le persone:", e)


def carica_persone():
    with galleria_lock:
        galleria.clear()
    nomi_identita.clear()
    p = percorso_persone()
    if p is None or not os.path.exists(p):
        return
    try:
        with open(p, "r", encoding="utf-8") as f:
            dati = json.load(f)
        with galleria_lock:
            for k, v in dati.get("galleria", {}).items():
                galleria[int(k)] = [np.asarray(c, dtype=np.float32) for c in v]
        for k, v in dati.get("nomi", {}).items():
            nomi_identita[int(k)] = v
        print(f"Caricate {len(galleria)} persone da {p}")
    except (OSError, ValueError) as e:
        print("File persone non leggibile:", e)


def fit_size(w, h):
    """Dimensioni che stanno in dim x dim mantenendo le proporzioni."""
    if w >= h:
        return dim, max(1, (dim * h) // w)
    return max(1, (dim * w) // h), dim


def colore_rgb(identita):
    if identita < 0:
        return COLORE_SCONOSCIUTO
    return COLORI[identita % len(COLORI)]


def colore_hex(identita):
    return "#%02x%02x%02x" % colore_rgb(identita)


def nome_identita(identita):
    if identita < 0:
        return "non riconosciuto"
    return nomi_identita.get(identita, f"persona {identita + 1}")


def rileva_volti(frame):
    """YOLO sul frame originale. Ritorna i box (x1, y1, x2, y2) ordinati da sinistra a destra."""
    frame_bgr = np.ascontiguousarray(frame[..., ::-1])
    res = model_face.predict(frame_bgr, **predict_kwargs)[0]
    return sorted(res.boxes.xyxy.cpu().numpy().tolist(), key=lambda b: b[0])


def calcola_embedding(frame, box):
    """Embedding ArcFace del volto dentro il box YOLO, oppure None se il volto è troppo
    piccolo o InsightFace non riesce ad allinearlo."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if min(w, h) < MIN_VOLTO_PX:
        return None

    h_img, w_img = frame.shape[:2]
    cx1 = int(max(0, x1 - w * MARGINE_CROP))
    cy1 = int(max(0, y1 - h * MARGINE_CROP))
    cx2 = int(min(w_img, x2 + w * MARGINE_CROP))
    cy2 = int(min(h_img, y2 + h * MARGINE_CROP))
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None

    ch, cw = crop.shape[:2]
    scala = EMB_SIZE / max(ch, cw)
    crop = Image.fromarray(crop).resize((max(1, int(cw * scala)), max(1, int(ch * scala))),
                                        Image.BILINEAR)
    crop_bgr = np.ascontiguousarray(np.asarray(crop)[..., ::-1])

    try:
        facce = face_app.get(crop_bgr)
    except Exception as e:
        print("Errore InsightFace:", e)
        return None
    facce = [f for f in facce if f.det_score >= MIN_DET_SCORE]
    if not facce:
        return None

    # il volto più grande nel ritaglio è quello del box
    f = max(facce, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return np.asarray(f.normed_embedding, dtype=np.float32)


def disegna_box(frame, coords, identita, pad=pixels):
    """Disegna i box espansi di `pad` pixel, ognuno col colore della sua identità."""
    img = Image.fromarray(frame).copy()     # copy: non modifica mai l'array originale
    draw = ImageDraw.Draw(img)
    h_img, w_img = frame.shape[:2]

    for (x1, y1, x2, y2), ident in zip(coords, identita):
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w_img - 1, x2 + pad)
        y2 = min(h_img - 1, y2 + pad)
        draw.rectangle([x1, y1, x2, y2], outline=colore_rgb(ident), width=line_width)

    return img

import subprocess
import time
import imageio_ffmpeg
from moviepy import VideoFileClip, VideoClip
from tkinter import ttk, HORIZONTAL, filedialog
# ---------------------------------------------------------------- capelli (SAM3)
def colora_capelli(img_rgb, mask, coords, identita,
                   gamma=0.65,      # <1 schiarisce le ombre (0.5 = molto, 1 = nulla)
                   forza_L=0.6,     # quanto avvicinare la luminosità a quella del colore target
                   contrasto=1.2,   # >1 enfatizza le ciocche
                   sat=0.9,         # saturazione del colore finale
                   raggio=1.8,      # max distanza dal volto, in multipli del lato del box
                   sfuma=1.5):
    """Ricolora i capelli per persona mantenendo texture e luci."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or not coords:
        return img_rgb

    # --- assegna ogni pixel al volto più vicino e scarta quelli troppo lontani ---
    migliore = np.full(len(xs), np.inf, dtype=np.float32)
    indice = np.zeros(len(xs), dtype=np.int64)
    for k, (x1, y1, x2, y2) in enumerate(coords):
        dx = np.maximum(np.maximum(x1 - xs, xs - x2), 0)
        dy = np.maximum(np.maximum(y1 - ys, ys - y2), 0)
        d = np.hypot(dx, dy).astype(np.float32)
        meglio = d < migliore
        migliore[meglio] = d[meglio]
        indice[meglio] = k
    lati = np.array([max(x2 - x1, y2 - y1) for x1, y1, x2, y2 in coords], dtype=np.float32)
    valido = migliore <= raggio * lati[indice]

    mask_pulita = np.zeros(mask.shape, dtype=np.float32)
    mask_pulita[ys[valido], xs[valido]] = 1.0
    ys, xs, indice = ys[valido], xs[valido], indice[valido]
    if len(xs) == 0:
        return img_rgb

    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_out = lab.copy()

    for k in np.unique(indice):
        sel = indice == k
        py, px = ys[sel], xs[sel]
        rgb_t = np.uint8([[colore_rgb(identita[k])]])
        Lt, at, bt = cv2.cvtColor(rgb_t, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
        Lt /= 255.0

        Ln = lab[py, px, 0] / 255.0
        Ln = Ln ** gamma                                  # schiarisce le ombre
        m = Ln.mean()
        L2 = m + contrasto * (Ln - m)                     # preserva/enfatizza la texture
        L3 = np.clip(L2 + forza_L * (Lt - m), 0.0, 1.0)   # avvicina al colore target

        # croma più debole dove è scuro (meno effetto "vernice")
        k_ch = np.clip(L3 / max(Lt, 1e-3), 0.55, 1.15)
        lab_out[py, px, 0] = L3 * 255.0
        lab_out[py, px, 1] = 128 + sat * (at - 128) * k_ch
        lab_out[py, px, 2] = 128 + sat * (bt - 128) * k_ch

    ricolorata = cv2.cvtColor(np.clip(lab_out, 0, 255).astype(np.uint8),
                              cv2.COLOR_LAB2RGB).astype(np.float32)

    # bordo sfumato: erodi di poco, poi sfuma
    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    alpha = cv2.erode(mask_pulita, nucleo)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sfuma)[..., None]

    out = img_rgb.astype(np.float32) * (1 - alpha) + ricolorata * alpha
    return out.astype(np.uint8)


sam_lock = threading.Lock()



from translate import Translator

# Cache della traduzione
PROMPT_TRADOTTO = None



def maschera_capelli(frame, separate=False):

    global PROMPT_TRADOTTO

    h, w = frame.shape[:2]
    mask_vuota = np.zeros((h, w), dtype=bool)

    try:
        # Traduzione prompt una sola volta
        if PROMPT_TRADOTTO is None:
            testo_ita = text.get("1.0", tk.END).strip()
            if not testo_ita:
                testo_ita = HAIR_PROMPT

            translator = Translator(from_lang="it", to_lang="en")
            risultato = translator.translate(testo_ita)
            PROMPT_TRADOTTO = risultato.strip() if risultato else HAIR_PROMPT

            print("Prompt SAM3:", PROMPT_TRADOTTO)

        with sam_lock:

            immagine = Image.fromarray(
                np.ascontiguousarray(frame)
            )

            stato = sam_processor.set_image(immagine)

            output = sam_processor.set_text_prompt(
                state=stato,
                prompt=PROMPT_TRADOTTO
            )

            scores = output["scores"].detach().reshape(-1)

            keep = torch.where(scores >= HAIR_SCORE)[0]

            if len(keep) == 0:
                return [] if separate else mask_vuota

            masks = output["masks"].detach()
            masks = masks.reshape(-1, *masks.shape[-2:])
            masks_np = masks[keep].float().cpu().numpy()

        maschere_individuali = []

        # Estrazione delle singole maschere
        for m in masks_np:

            m = m > 0.5

            if m.shape != (h, w):
                m = np.asarray(
                    Image.fromarray(
                        m.astype(np.uint8) * 255
                    ).resize((w, h), Image.NEAREST)
                ) > 127

            # Dilatazione individuale
            if HAIR_DILATE_PX > 0:
                lato = 2 * HAIR_DILATE_PX + 1
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (lato, lato)
                )
                m = cv2.dilate(
                    m.astype(np.uint8),
                    kernel
                ) > 0

            if not m.any():
                continue

            # Elimina maschere quasi identiche
            duplicata = False

            for precedente in maschere_individuali:

                intersezione = np.logical_and(m, precedente).sum()
                unione = np.logical_or(m, precedente).sum()

                iou = intersezione / max(unione, 1)

                if iou > 0.75:
                    duplicata = True
                    break

            if not duplicata:
                maschere_individuali.append(m)

        if separate:
            return maschere_individuali

        # Modalità normale: unione per anteprima
        for m in maschere_individuali:
            mask_vuota |= m

        return mask_vuota

    except Exception as e:
        print("Errore SAM3:", e)
        return [] if separate else mask_vuota





def rilevamento_capelli(frame, annotated, coords, identita):
    """Preview: ricolora i capelli per persona e disegna i box sopra."""
    if not capelli_attivi:
        return annotated
    try:
        mask = maschera_capelli(frame)
        if not mask.any():
            print(f"Nessun capello femminile rilevato (soglia {HAIR_SCORE}).")
            return annotated
        ricolorato = colora_capelli(frame, mask, coords, identita)
        return disegna_box(ricolorato, coords, identita)
    except Exception as e:
        print("Errore rilevamento capelli:", e)
        return annotated


# ---------------------------------------------------------------- export maschere
esportazione_in_corso = False
stop_esportazione = False
testo_progresso = None          # scritto dal thread, letto dal thread principale
progresso_i = 0
progresso_tot = 1
progresso_t0 = 0.0


class EsportazioneInterrotta(Exception):
    pass


def nvenc_disponibile():
    """True se l'ffmpeg usato da moviepy ha l'encoder NVIDIA h264_nvenc."""
    try:
        out = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=10).stdout
        return "h264_nvenc" in out
    except Exception:
        return False




def _esporta_worker(percorso_out, clip, fps, totale):

    global esportazione_in_corso, testo_progresso
    global progresso_i, progresso_tot, progresso_t0

    progresso_i = 0
    progresso_tot = totale
    progresso_t0 = time.time()

    try:

        with video_lock:
            w0, h0 = clip.size

        # Ridimensionamento proporzionale
        nuova_w = 720
        nuova_h = int(round(h0 * nuova_w / w0))
        nuova_h -= nuova_h % 2

        print(f"Esportazione: {nuova_w}x{nuova_h}")

        # Tracce individuali
        tracks = {}
        prossimo_id = 0

        zero = np.zeros((nuova_h, nuova_w), dtype=bool)
        zero_packed = np.packbits(zero.ravel())

        # ---------------------------------------
        # FASE 1: RILEVAMENTO E TRACKING
        # ---------------------------------------

        for i in range(totale):

            if stop_esportazione or not running or video is not clip:
                raise EsportazioneInterrotta()

            with video_lock:
                frame = clip.get_frame(i / fps).copy()

            frame_piccolo = cv2.resize(
                frame,
                (nuova_w, nuova_h),
                interpolation=cv2.INTER_AREA
            )

            # Restituisce le maschere separate
            maschere = maschera_capelli(
                frame_piccolo,
                separate=True
            )

            # Aggiunge un frame nero alle tracce esistenti
            for traccia in tracks.values():
                traccia["frames"].append(zero_packed)

            # -----------------------------------
            # ABBINAMENTO TRAMITE IoU
            # -----------------------------------

            coppie = []

            for j, m in enumerate(maschere):

                for ident, traccia in tracks.items():

                    precedente = traccia["last"]

                    inter = np.logical_and(m, precedente).sum()
                    union = np.logical_or(m, precedente).sum()

                    iou = inter / max(union, 1)

                    if iou >= 0.03:
                        coppie.append((iou, j, ident))

            # Prima gli abbinamenti migliori
            coppie.sort(reverse=True)

            usate_maschere = set()
            usate_tracce = set()

            for iou, j, ident in coppie:

                if j in usate_maschere:
                    continue

                if ident in usate_tracce:
                    continue

                traccia = tracks[ident]

                traccia["frames"][-1] = np.packbits(
                    maschere[j].ravel()
                )

                traccia["last"] = maschere[j]
                traccia["count"] += 1

                usate_maschere.add(j)
                usate_tracce.add(ident)

            # Nuove tracce per maschere non abbinate
            for j, m in enumerate(maschere):

                if j in usate_maschere:
                    continue

                frames_iniziali = [
                    zero_packed
                ] * i

                frames_iniziali.append(
                    np.packbits(m.ravel())
                )

                tracks[prossimo_id] = {
                    "last": m,
                    "frames": frames_iniziali,
                    "count": 1
                }

                prossimo_id += 1

            progresso_i = i + 1

        # ---------------------------------------
        # FASE 2: ESPORTAZIONE VIDEO SEPARATI
        # ---------------------------------------

        testo_progresso = "Scrittura video individuali..."

        # Scarta tracce troppo brevi
        tracks = {
            ident: traccia
            for ident, traccia in tracks.items()
            if traccia["count"] >= 3
        }

        if not tracks:
            testo_progresso = "Nessuna maschera capelli valida."
            return

        base = os.path.splitext(percorso_out)[0]

        nvenc = nvenc_disponibile()

        for numero, (ident, traccia) in enumerate(
            sorted(tracks.items()), start=1
        ):

            if stop_esportazione:
                raise EsportazioneInterrotta()

            frames = traccia["frames"]

            percorso_video = (
                f"{base}_capelli_{numero}.mp4"
            )

            def frame_da_array(t, frames=frames):

                idx = min(
                    int(round(t * fps)),
                    len(frames) - 1
                )

                bits = np.unpackbits(
                    frames[idx],
                    count=nuova_h * nuova_w
                ).reshape(nuova_h, nuova_w)

                grigio = (bits * 255).astype(np.uint8)

                return np.repeat(
                    grigio[:, :, None],
                    3,
                    axis=2
                )

            video_mask = VideoClip(
                frame_da_array,
                duration=totale / fps
            )

            print(
                f"Esporto capelli {numero}: {percorso_video}"
            )

            if nvenc:

                video_mask.write_videofile(
                    percorso_video,
                    fps=fps,
                    codec="h264_nvenc",
                    audio=False,
                    preset="p5",
                    logger=None,
                    ffmpeg_params=[
                        "-rc", "vbr",
                        "-cq", "10",
                        "-b:v", "0"
                    ]
                )

            else:

                video_mask.write_videofile(
                    percorso_video,
                    fps=fps,
                    codec="libx264",
                    audio=False,
                    preset="medium",
                    logger=None,
                    ffmpeg_params=["-crf", "8"]
                )

            video_mask.close()

            if stop_esportazione:
                raise EsportazioneInterrotta()

        testo_progresso = (
            f"Completato: {len(tracks)} video di maschere capelli."
        )

        print(testo_progresso)

    except EsportazioneInterrotta:

        testo_progresso = "Esportazione interrotta"

    except Exception as e:

        print("Errore esportazione:", e)
        testo_progresso = "Errore esportazione (vedi console)"

    finally:

        esportazione_in_corso = False




def esporta_maschere():
    global esportazione_in_corso, stop_esportazione, testo_progresso, progresso_i
    if video is None or esportazione_in_corso:
        return
    iniziale = os.path.splitext(os.path.basename(path_video))[0] + "_mask.mp4"
    percorso_out = filedialog.asksaveasfilename(
        defaultextension=".mp4", initialfile=iniziale, filetypes=[("MP4", "*.mp4")])
    if not percorso_out:
        return
    esportazione_in_corso = True
    stop_esportazione = False
    progresso_i = 0
    testo_progresso = "Avvio esportazione..."
    threading.Thread(target=_esporta_worker,
                     args=(percorso_out, video, video.fps, n_frames),
                     daemon=True).start()
    aggiorna_export()


def aggiorna_export():
    """Thread principale: barra di progresso, stima del tempo, pulsante."""
    if esportazione_in_corso and 0 < progresso_i < progresso_tot:
        trascorso = time.time() - progresso_t0
        rimasto = trascorso / progresso_i * (progresso_tot - progresso_i)
        m, s = divmod(int(rimasto), 60)
        h, m = divmod(m, 60)
        label_export.config(
            text=f"{progresso_i} / {progresso_tot}  "
                 f"({100 * progresso_i // progresso_tot}%)  "
                 f"restano ~{h}h {m:02d}m {s:02d}s")
        barra_export.config(maximum=progresso_tot, value=progresso_i)
    elif testo_progresso:
        label_export.config(text=testo_progresso)

    if esportazione_in_corso:
        b_esporta.config(text="Ferma export", command=ferma_export)
        window.after(300, aggiorna_export)
    else:
        b_esporta.config(text="Esporta maschere", command=esporta_maschere)


def ferma_export():
    global stop_esportazione
    stop_esportazione = True


# ---------------------------------------------------------------- worker (thread)
def worker():
    """Thread unico: legge l'ultimo frame richiesto, rileva e riconosce i volti
    (e, se attivo, i capelli femminili). Non tocca mai i widget tkinter."""
    while running:
        request_event.wait()
        if not running:
            break
        request_event.clear()
        index = requested_index

        # 1) lettura del frame
        with video_lock:
            if video is None:
                continue
            index = max(0, min(index, n_frames - 1))
            try:
                frame = video.get_frame(index / video.fps).copy()   # array RGB
            except Exception as e:
                print("Errore lettura frame:", e)
                continue

        # 2) rilevamento, riconoscimento e disegno dei box
        try:
            coords = rileva_volti(frame)
            embs = [calcola_embedding(frame, b) for b in coords]
            identita = assegna_identita(embs, impara=True)
            annotated = disegna_box(frame, coords, identita)
        except Exception as e:
            print("Errore rilevamento/riconoscimento:", e)
            continue

        # 3) capelli femminili (SAM3), solo se la casella e' attiva
        annotated = rilevamento_capelli(frame, annotated, coords, identita)

        # 4) ridimensionamento per le canvas
        img = Image.fromarray(frame)
        size = fit_size(*img.size)
        img = img.resize(size, Image.BILINEAR)
        annotated = annotated.resize(size, Image.BILINEAR)

        results.put((index, img, annotated, identita, frame, coords, embs))

# ---------------------------------------------------------------- righe "persona" (thread principale)
def crea_riga():
    """Crea i widget di una riga: nome modificabile, colore, pulsante Elimina."""
    riga = tk.Frame(frame_risult, bg="light gray")

    entry = tk.Entry(riga, font=("Arial", 10), width=15)
    entry.grid(row=0, column=0, padx=5, pady=2)

    color = tk.Canvas(riga, width=20, height=20, bg="red", highlightthickness=0)
    color.grid(row=0, column=1, padx=5, pady=2)

    elemento = {"frame": riga, "entry": entry, "color": color, "testo": None}

    bottone = tk.Button(riga, text="Elimina", font=("Arial", 10), fg="red", bg="white",
                        command=lambda e=elemento: elimina_riga(e))
    bottone.grid(row=0, column=2, padx=5, pady=2)
    elemento["button"] = bottone

    # "persona N" si applica subito mentre scrivi; un nome libero con Invio o cliccando altrove
    entry.bind("<KeyRelease>", lambda ev, e=elemento: applica_modifica(e, False))
    entry.bind("<Return>", lambda ev, e=elemento: applica_modifica(e, True))
    entry.bind("<FocusOut>", lambda ev, e=elemento: applica_modifica(e, True))

    return elemento


def aggiorna_righe(identita):
    """Allinea le righe ai volti del frame: una riga per volto, da sinistra a destra."""
    righe = Raccoglitore_elementi_riconoscimento_facciale

    while len(righe) < len(identita):
        righe.append(crea_riga())

    for i, el in enumerate(righe):
        if i < len(identita):
            ident = identita[i]
            el["color"].config(bg=colore_hex(ident))

            testo = nome_identita(ident)
            # non sovrascrivere il testo mentre l'utente lo sta scrivendo
            if el["testo"] != testo and window.focus_get() is not el["entry"]:
                el["entry"].delete(0, tk.END)
                el["entry"].insert(0, testo)
                el["testo"] = testo

            el["frame"].grid(row=i, column=0, sticky="w")
        else:
            el["frame"].grid_remove()


def ridisegna_output():
    """Ricalcola le identità del frame mostrato con la galleria aggiornata e ridisegna
    subito la canvas di output e le righe, senza rieseguire YOLO."""
    global photo_out, identita_ora
    if ultimo_frame is None:
        return

    identita = assegna_identita(ultimo_embs, impara=False)
    identita_ora = identita

    annotated = disegna_box(ultimo_frame, ultimo_coords, identita)
    annotated = annotated.resize(fit_size(*annotated.size), Image.BILINEAR)
    photo_out = ImageTk.PhotoImage(annotated)

    canvasOut.delete("all")
    canvasOut.create_image(dim // 2, dim // 2, image=photo_out)
    aggiorna_righe(identita)


def applica_modifica(elemento, definitivo):
    """Chiamata dal campo nome della riga.
    - 'persona N' -> correzione: il volto di quella riga è la persona N. Il sistema lo impara
                     (galleria) e da quel momento la riconosce anche nei frame successivi.
                     Se un altro volto del frame era già la persona N, le due si scambiano.
    - altro testo -> nome personalizzato per quella persona (solo con Invio / focus perso)."""
    testo = elemento["entry"].get().strip()
    if testo == elemento["testo"]:
        return                                   # nulla è cambiato

    righe = Raccoglitore_elementi_riconoscimento_facciale
    if elemento not in righe:
        return
    slot = righe.index(elemento)
    if slot >= len(identita_ora) or slot >= len(ultimo_embs):
        return

    corrisponde = re.fullmatch(r"persona\s*(\d+)", testo, re.IGNORECASE)
    if not corrisponde and not definitivo:
        return                                   # sta ancora scrivendo un nome libero

    vecchia = identita_ora[slot]
    if vecchia < 0:
        print("Volto senza riconoscimento (troppo piccolo o sfocato): impossibile correggerlo.")
        elemento["testo"] = None                 # il testo verrà riscritto
        return

    modificato = False
    if corrisponde:
        nuova = int(corrisponde.group(1)) - 1
        emb = ultimo_embs[slot]
        if nuova >= 0 and nuova != vecchia and emb is not None:
            correzioni = [(emb, vecchia, nuova)]
            # se un altro volto del frame era già la persona N, si scambiano
            for j, ident in enumerate(identita_ora):
                if j != slot and ident == nuova and ultimo_embs[j] is not None:
                    correzioni.append((ultimo_embs[j], nuova, vecchia))
            riassegna_volti(correzioni)
            modificato = True
    elif testo:
        nomi_identita[vecchia] = testo
        modificato = True

    elemento["testo"] = testo

    if definitivo:
        window.focus_set()              # toglie il focus: il testo verrà riscritto in forma standard

    if modificato:
        ridisegna_output()
        salva_persone()
        print("Persone note:", sorted(i + 1 for i in galleria))


def elimina_riga(elemento):
    elemento["frame"].destroy()
    Raccoglitore_elementi_riconoscimento_facciale.remove(elemento)
    aggiorna_righe(identita_ora)


# ---------------------------------------------------------------- thread principale
def poll_results():
    """Gira nel thread principale: mostra l'ultimo frame pronto."""
    global photo_input, photo_out, identita_ora, ultimo_frame, ultimo_coords, ultimo_embs
    last = None
    try:
        while True:
            last = results.get_nowait()   # scarta i frame vecchi
    except queue.Empty:
        pass

    if last is not None:
        index, img, annotated, identita, frame, coords, embs = last
        ultimo_frame = frame
        ultimo_coords = coords
        ultimo_embs = embs
        identita_ora = identita

        photo_input = ImageTk.PhotoImage(img)
        photo_out = ImageTk.PhotoImage(annotated)

        canvasInput.delete("all")
        canvasInput.create_image(dim // 2, dim // 2, image=photo_input)
        canvasOut.delete("all")
        canvasOut.create_image(dim // 2, dim // 2, image=photo_out)
        label_frame.config(text=f"Frame {index + 1} / {n_frames}")

        aggiorna_righe(identita)

    if running:
        window.after(15, poll_results)


def request_frame(index):
    global requested_index
    requested_index = index
    request_event.set()


def on_drop(event):
    global path_video, n_frames, video

    files = window.tk.splitlist(event.data)
    nuovo_path = files[0]

    try:
        new_video = VideoFileClip(nuovo_path)
    except Exception as e:
        print("Impossibile aprire il video:", e)
        return

    salva_persone()                      # salva le persone del video precedente
    path_video = nuovo_path

    with video_lock:
        if video is not None:
            video.close()
        video = new_video
        n_frames = int(video.duration * video.fps)

    carica_persone()                     # se il video è già stato analizzato, le persone sono note

    scorriframes.config(from_=0, to=n_frames - 1)
    scorriframes.set(0)
    request_frame(0)


def f_scroll(value):
    request_frame(int(float(value)))


def step_frame(delta):
    if video is None:
        return
    new = int(float(scorriframes.get())) + delta
    new = max(0, min(new, n_frames - 1))
    scorriframes.set(new)
    request_frame(new)


def toggle_capelli():
    global capelli_attivi
    capelli_attivi = bool(var_capelli.get())
    if video is not None:
        request_frame(int(float(scorriframes.get())))   # ricalcola il frame mostrato


def on_close():
    global running
    running = False
    request_event.set()
    salva_persone()
    with video_lock:
        if video is not None:
            video.close()
    window.destroy()


# ---------------------------------------------------------------- interfaccia
window = TkinterDnD.Tk()
window.geometry("1300x700")
window.config(background='light blue')
window.resizable(False, False)
window.protocol("WM_DELETE_WINDOW", on_close)

frame_canvas = tk.Frame(window, bg="gray")
frame_canvas.grid(row=0, column=0)

canvasInput = tk.Canvas(frame_canvas, width=dim, height=dim, bg="red")
canvasInput.grid(row=0, column=0)
canvasInput.create_text(dim // 2, dim // 2, text="Trascina qui un file video",
                        font=("Arial", 15), fill="white")
canvasInput.drop_target_register(DND_FILES)
canvasInput.dnd_bind("<<Drop>>", on_drop)

canvasOut = tk.Canvas(frame_canvas, width=dim, height=dim, bg="dark green")
canvasOut.grid(row=0, column=1)
canvasOut.create_text(dim // 2, dim // 2, text="FRAME OUTPUT",
                      font=("Arial", 15), fill="white")

frame_risult = tk.Frame(window, bg="light gray")
frame_risult.grid(row=0, column=2, sticky="nw", padx=10, pady=2)

frame_strumenti = tk.Frame(window, bg="gray")
frame_strumenti.grid(row=1, column=0, sticky="ew")

scorriframes = ttk.Scale(frame_strumenti, from_=0, to=100, orient=HORIZONTAL,
                         length=300, command=f_scroll)
scorriframes.grid(row=0, column=0, padx=10, pady=10)

label_frame = tk.Label(frame_strumenti, text="Frame 0 / 0", bg="gray", fg="white")
label_frame.grid(row=0, column=1, padx=10)

b_piu = tk.Button(frame_strumenti, text="+", bg="light green", fg="blue",
                  command=lambda: step_frame(1))
b_piu.grid(row=0, column=2, padx=10)

b_meno = tk.Button(frame_strumenti, text="-", bg="green", fg="white",
                   command=lambda: step_frame(-1))
b_meno.grid(row=0, column=3, padx=10)

var_capelli = tk.BooleanVar(value=False)
c_capelli = tk.Checkbutton(frame_strumenti, text="Capelli", variable=var_capelli,
                           command=toggle_capelli, bg="gray", fg="white",
                           selectcolor="dark gray", activebackground="gray")
c_capelli.grid(row=0, column=4, padx=10)


# Pulsante esportazione
b_esporta = tk.Button(
    frame_strumenti,
    text="Esporta maschere",
    command=esporta_maschere
)
b_esporta.grid(row=0, column=5, padx=10)

# Frame esportazione
frame_export = tk.Frame(window)
frame_export.grid(
    row=2,
    column=0,
    sticky="ew",
    padx=10,
    pady=5
)

# Barra progresso
barra_export = ttk.Progressbar(
    frame_export,
    orient=HORIZONTAL,
    length=500,
    mode="determinate"
)
barra_export.pack(side="left", padx=(0, 10))

# Etichetta progresso
label_export = tk.Label(
    frame_export,
    text=""
)
label_export.pack(side="left")

# Area testo
text = tk.Text(
    window,
    width=20,
    height=7)
text.grid(
    row=3,
    column=0,
    padx=10,
    pady=5,
    sticky="ew"
)
text.insert('1.0',HAIR_PROMPT)


# legenda capelli: etichetta + quadratino dello stesso colore del volto della persona N
frame_capelli = tk.Frame(window, bg="light gray")
frame_capelli.grid(row=1, column=2, sticky="nw", padx=10, pady=2)

for n in (1, 2):
    etichetta = tk.Label(frame_capelli, text=f"hair persona {n}", font=("Arial", 10),
                         bg="light gray", width=15, anchor="w")
    etichetta.grid(row=n - 1, column=0, padx=5, pady=2)
    quadrato = tk.Canvas(frame_capelli, width=20, height=20, bg=colore_hex(n - 1),
                         highlightthickness=0)
    quadrato.grid(row=n - 1, column=1, padx=5, pady=2)

# ---------------------------------------------------------------- avvio
threading.Thread(target=worker, daemon=True).start()
poll_results()
window.mainloop()