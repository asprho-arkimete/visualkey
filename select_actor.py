import os
import subprocess
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from tkinterdnd2 import TkinterDnD, DND_FILES
from moviepy import VideoFileClip  # moviepy 2.x (con la 1.x: from moviepy.editor import VideoFileClip)
from PIL import Image, ImageTk, ImageDraw
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import imageio_ffmpeg  # installato insieme a moviepy

# ---------- impostazioni ----------
MAX_SIZE = 512
MODEL_PATH = "yolov12l-face.pt"
CONFIDENZA = 0.4
SOGLIA = 0.40          # somiglianza coseno minima per dire "è la stessa persona"
PASSO_SCAN = 1.0       # secondi tra un frame e l'altro durante la scansione
INCLINAZIONE = 0.04    # 0 = taglio verticale; 0.04 = diagonale
COLORI = ["lime", "deepskyblue", "yellow", "red"]   # persona 1, 2, 3, 4
COLORE_EXTRA = "white"

model = YOLO(MODEL_PATH)

app = FaceAnalysis(name="buffalo_l",
                   providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(320, 320))

window = TkinterDnD.Tk()
window.title("Select Actor")
window.geometry("1024x900")
window.config(background="gray")

video = None
foto = None
identita = []   # archivio: [{"emb": vettore normalizzato, "n": numero di campioni}, ...]


# ---------- riconoscimento ----------
def embedding_viso(img_bgr, box):
    """Ritaglia il viso (con un po' di margine) e ne calcola l'embedding."""
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * 0.3, (y2 - y1) * 0.3
    H, W = img_bgr.shape[:2]
    crop = img_bgr[max(0, int(y1 - my)):min(H, int(y2 + my)),
                   max(0, int(x1 - mx)):min(W, int(x2 + mx))]
    if crop.size == 0:
        return None
    facce = app.get(crop)
    if not facce:
        return None
    f = max(facce, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return f.normed_embedding


def limite_persone():
    """Legge il numero massimo di persone dal campo di testo (0 = nessun limite)."""
    try:
        return max(0, int(entry_limite.get()))
    except ValueError:
        return 0


def assegna_identita(embs, apprendi=True):
    """Restituisce per ogni embedding l'indice di identità.
    Con un limite di persone: oltre il limite non si creano nuove identità e i
    volti vengono assegnati a quella più simile, anche sotto soglia."""
    risultato = [None] * len(embs)
    limite = limite_persone()

    coppie = []
    for i, e in enumerate(embs):
        if e is None:
            continue
        for j, idn in enumerate(identita):
            coppie.append((float(np.dot(e, idn["emb"])), i, j))
    coppie.sort(reverse=True)

    # 1° passaggio: corrispondenze sopra soglia
    usati_viso, usate_idn = set(), set()
    for sim, i, j in coppie:
        if sim < SOGLIA:
            break
        if i in usati_viso or j in usate_idn:
            continue
        risultato[i] = j
        usati_viso.add(i)
        usate_idn.add(j)
        if apprendi:
            idn = identita[j]
            nuovo = idn["emb"] * idn["n"] + embs[i]
            idn["emb"] = nuovo / np.linalg.norm(nuovo)
            idn["n"] += 1

    # 2° passaggio: nuove identità, solo se non si è raggiunto il limite
    if apprendi:
        for i, e in enumerate(embs):
            if e is not None and risultato[i] is None:
                if limite and len(identita) >= limite:
                    break
                identita.append({"emb": e.copy(), "n": 1})
                risultato[i] = len(identita) - 1
                usati_viso.add(i)
                usate_idn.add(len(identita) - 1)

    # 3° passaggio: con un limite, i volti rimasti vanno all'identità più simile
    # (senza aggiornare l'archivio, per non "sporcarlo" con match incerti)
    if limite:
        for sim, i, j in coppie:
            if i in usati_viso or j in usate_idn:
                continue
            risultato[i] = j
            usati_viso.add(i)
            usate_idn.add(j)
    return risultato


def rileva(img_rgb, apprendi=True):
    """Restituisce [(box, indice_identità), ...] per il frame."""
    results = model.predict(img_rgb, conf=CONFIDENZA, verbose=False)
    boxes = [b.xyxy[0].tolist() for r in results for b in r.boxes]
    boxes.sort(key=lambda b: b[0])

    img_bgr = np.ascontiguousarray(np.array(img_rgb)[:, :, ::-1])
    embs = [embedding_viso(img_bgr, b) for b in boxes]
    ids = assegna_identita(embs, apprendi)
    return list(zip(boxes, ids))


# ---------- ritaglio a fasce con alpha ----------
def persone_selezionate():
    var = (rileva_viso1, rileva_viso2, rileva_viso3, rileva_viso4)
    return {i for i, v in enumerate(var) if v.get()}


def poligono(xl, xr, W, H):
    """Fascia tra due tagli (leggermente inclinati). I bordi del frame restano dritti."""
    s = int(W * INCLINAZIONE)
    tl, bl = (xl + s, xl - s) if xl > 0 else (0, 0)
    tr, br = (xr + s, xr - s) if xr < W else (W, W)
    return [(tl, 0), (tr, 0), (br, H), (bl, H)]


def ritaglia_persone(img, visi, persone):
    """RGBA: visibile solo la fascia delle persone indicate, il resto trasparente."""
    W, H = img.size
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)

    # tutti i visi (anche non selezionati) fanno da separatori
    ordinati = sorted(visi, key=lambda v: (v[0][0] + v[0][2]) / 2)
    centri = [(b[0] + b[2]) / 2 for b, _ in ordinati]

    for k, (_, idx) in enumerate(ordinati):
        if idx not in persone:
            continue
        xl = 0 if k == 0 else (centri[k - 1] + centri[k]) / 2
        xr = W if k == len(ordinati) - 1 else (centri[k] + centri[k + 1]) / 2
        draw.polygon(poligono(xl, xr, W, H), fill=255)

    arr = np.array(img.convert("RGBA"))
    m = np.array(mask)
    arr[m == 0, :3] = 0      # fuori dalla fascia: RGB nero, non solo trasparente
    arr[..., 3] = m
    return Image.fromarray(arr, "RGBA")


def scacchiera(w, h, q=24):
    y, x = np.indices((h, w))
    v = np.where(((x // q) + (y // q)) % 2 == 0, 90, 60).astype(np.uint8)
    return Image.fromarray(np.dstack([v, v, v]))


def disegna_visi(img):
    """Anteprima: con checkbox attive mostra la fascia scelta su scacchiera (= trasparenza)."""
    visi = rileva(img)
    sel = persone_selezionate()

    if sel:
        rgba = ritaglia_persone(img, visi, sel)
        base = scacchiera(*img.size)
        base.paste(rgba, (0, 0), rgba)
        img = base

    spessore = max(2, max(img.size) // 250)
    draw = ImageDraw.Draw(img)
    for (x1, y1, x2, y2), idx in visi:
        if sel and idx not in sel:
            continue
        colore = "gray" if idx is None else (COLORI[idx] if idx < len(COLORI) else COLORE_EXTRA)
        draw.rectangle((x1, y1, x2, y2), outline=colore, width=spessore)
        if idx is not None:
            draw.text((x1 + 4, y1 + 4), f"P{idx + 1}", fill=colore)
    return img


def mostra_frame():
    global foto
    if video is None:
        return
    n = int(float(scorri_frames.get()))
    t = min(n / video.fps, video.duration - 0.001)

    frame_or = Image.fromarray(video.get_frame(t)).convert("RGB")
    frame_or = disegna_visi(frame_or)

    w, h = frame_or.size
    scala = MAX_SIZE / max(w, h)
    frame_ris = frame_or.resize((int(w * scala), int(h * scala)), Image.LANCZOS)

    foto = ImageTk.PhotoImage(frame_ris)
    frame.delete("all")
    frame.create_image(MAX_SIZE // 2, MAX_SIZE // 2, image=foto)


# ---------- scansione ed esportazione ----------
def scansiona_video():
    """Scorre il video e numera le persone in ordine di prima comparsa."""
    if video is None:
        return
    identita.clear()
    t = 0.0
    while t < video.duration:
        rileva(Image.fromarray(video.get_frame(t)).convert("RGB"))
        label_stato.config(text=f"Scansione: {int(t / video.duration * 100)}%  "
                                f"persone trovate: {len(identita)}")
        window.update()
        t += PASSO_SCAN
    label_stato.config(text=f"Scansione finita: {len(identita)} persone")
    mostra_frame()


def reset_identita():
    identita.clear()
    label_stato.config(text="Archivio svuotato")
    mostra_frame()


def esporta_alpha():
    """Un file .mov ProRes 4444 con canale alpha per ogni persona spuntata."""
    if video is None:
        return
    sel = persone_selezionate()
    if not sel:
        label_stato.config(text="Spunta almeno una persona")
        return
    if not identita:
        label_stato.config(text="Premi prima 'Scansiona video'")
        return
    cartella = filedialog.askdirectory()
    if not cartella:
        return

    W, H = video.size
    fps = video.fps
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    procs = {}
    for p in sorted(sel):
        cmd = [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
               "-r", str(fps), "-i", "-", "-c:v", "prores_ks", "-profile:v", "4444",
               "-pix_fmt", "yuva444p10le", os.path.join(cartella, f"persona{p + 1}.mov")]
        procs[p] = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    totale = int(video.duration * fps)
    try:
        for i, fr in enumerate(video.iter_frames(fps=fps, dtype="uint8")):
            img = Image.fromarray(fr).convert("RGB")
            visi = rileva(img, apprendi=False)   # non altera la numerazione delle persone
            for p, pr in procs.items():
                pr.stdin.write(ritaglia_persone(img, visi, {p}).tobytes())
            if i % 10 == 0:
                label_stato.config(text=f"Esportazione: {i}/{totale}")
                window.update()
        esito = f"Esportati {len(procs)} file in {cartella}"
    except (BrokenPipeError, OSError):
        esito = "Errore ffmpeg: esportazione interrotta"
    finally:
        for pr in procs.values():
            try:
                pr.stdin.close()
            except OSError:
                pass
            pr.wait()
    label_stato.config(text=esito)


# ---------- interfaccia ----------
def f_drag_e_drop(event):
    global video
    path = window.tk.splitlist(event.data)[0]
    if video is not None:
        video.close()
    video = VideoFileClip(path)
    identita.clear()

    totale = int(video.duration * video.fps) - 1
    scorri_frames.config(from_=0, to=totale)
    scorri_frames.set(0)
    f_scorri_frames(0)


def f_scorri_frames(valore=None):
    label_frames.config(text=f"Frames: {int(float(scorri_frames.get()))}")
    mostra_frame()


frame_canvas = tk.Frame(window)
frame_canvas.grid(row=0, column=0, sticky="nw")
frame = tk.Canvas(frame_canvas, width=MAX_SIZE, height=MAX_SIZE, bg="red")
frame.create_text(MAX_SIZE // 2, MAX_SIZE // 2, text="Trascina qui il video", anchor="center")
frame.grid(row=0, column=0, sticky="nw")
frame.drop_target_register(DND_FILES)
frame.dnd_bind("<<Drop>>", f_drag_e_drop)

frame_strumenti = tk.Frame(window)
frame_strumenti.grid(row=1, column=0, sticky="nw")

label_frames = tk.Label(frame_strumenti, text="Frames")
label_frames.grid(row=0, column=0, sticky="nw")
scorri_frames = ttk.Scale(frame_strumenti, from_=0, to=100, orient="horizontal",
                          length=300, command=f_scorri_frames)
scorri_frames.grid(row=1, column=0, sticky="nw")

ttk.Button(frame_strumenti, text="Scansiona video",
           command=scansiona_video).grid(row=2, column=0, sticky="nw", pady=4)
ttk.Button(frame_strumenti, text="Reset persone",
           command=reset_identita).grid(row=3, column=0, sticky="nw")
ttk.Button(frame_strumenti, text="Esporta clip con alpha",
           command=esporta_alpha).grid(row=4, column=0, sticky="nw", pady=4)
label_stato = tk.Label(frame_strumenti, text="")
label_stato.grid(row=5, column=0, sticky="nw")

frame_strumenti2 = tk.Frame(window)
frame_strumenti2.grid(row=0, column=1, sticky="nw")
rileva_viso1 = tk.IntVar()
rileva_viso2 = tk.IntVar()
rileva_viso3 = tk.IntVar()
rileva_viso4 = tk.IntVar()
for riga, (testo, var) in enumerate([("persona 1", rileva_viso1), ("persona 2", rileva_viso2),
                                      ("persona 3", rileva_viso3), ("persona 4", rileva_viso4)]):
    tk.Checkbutton(frame_strumenti2, text=testo, variable=var,
                   command=mostra_frame).grid(row=riga, column=0, sticky="nw")
tk.Label(frame_strumenti2, text="Numero persone (0 = libero):").grid(row=4, column=0, sticky="nw", pady=(10, 0))

entry_limite = tk.Entry(frame_strumenti2, width=5)
entry_limite.insert(0, "2")
entry_limite.grid(row=5, column=0, sticky="nw")

window.mainloop()