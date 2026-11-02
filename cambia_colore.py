import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, colorchooser, ttk, messagebox
from moviepy import VideoFileClip, VideoClip   # con moviepy 1.x: from moviepy.editor import VideoFileClip, VideoClip
from PIL import Image, ImageTk, ImageFilter

try:
    import cv2                      # opzionale: molto più veloce
except ImportError:
    cv2 = None

window = tk.Tk()
window.geometry("1024x800")
window.config(background='gray')
window.title("change color")
window.resizable(False, False)

path_video = None
path_mask = None
video = None
video_mask = None

COLORE_ORIGINE = (0, 0, 0)        # nero
COLORE_NUOVO = (255, 0, 0)        # rosso
DIST_MAX = (3 * 255 ** 2) ** 0.5  # distanza massima nello spazio RGB (~441.7)

frame_originale = None            # ultimo frame ridimensionato, SENZA modifiche
contagocce_attivo = False
in_render = False
annulla_render = False


def _hex(rgb):
    return '#%02x%02x%02x' % tuple(int(c) for c in rgb)


def _tempo(clip, index):
    """Converte l'indice di frame in secondi, restando dentro la durata del clip."""
    return min(index / clip.fps, max(clip.duration - 1 / clip.fps, 0))


def _lum(rgb):
    """Luminosità percepita 0..1 (Rec. 709)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


def leggi_parametri():
    """Fotografia dei valori correnti di colori e slider."""
    return {
        'origine': COLORE_ORIGINE,
        'nuovo': COLORE_NUOVO,
        'tol_or': tolleranza_or.get(),
        'sfuma': tolleranza_camb.get(),
        'neri': tolleranza_neri.get(),
        'bordi': slider_bordi.get(),
    }


def _bordi(maschera, px):
    """Espande (px > 0) o restringe (px < 0) il bianco di una maschera booleana."""
    r = int(round(abs(px)))
    if r == 0:
        return maschera
    m = maschera.astype(np.uint8) * 255
    if cv2 is not None:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        m = cv2.dilate(m, k) if px > 0 else cv2.erode(m, k)
    else:                                   # fallback senza OpenCV: passi da 1 px
        img = Image.fromarray(m)
        filtro = ImageFilter.MaxFilter(3) if px > 0 else ImageFilter.MinFilter(3)
        for _ in range(r):
            img = img.filter(filtro)
        m = np.asarray(img)
    return m > 127


def _zona(t, size, bordi=0):
    """Maschera booleana (H, W) per il tempo t, con bordi espansi/ridotti di `bordi` px."""
    if video_mask is None:
        return np.ones((size[1], size[0]), dtype=bool)
    tm = min(t, max(video_mask.duration - 1 / video_mask.fps, 0))
    m = Image.fromarray(video_mask.get_frame(tm)).convert('L')
    m = m.resize(size, Image.Resampling.NEAREST)
    return _bordi(np.asarray(m) > 127, bordi)


def elabora(arr, zona, par, L_alta=None):
    """
    Cambia il colore dei capelli su un frame di qualsiasi risoluzione.
    arr  : array uint8 (H, W, 3)
    zona : maschera booleana (H, W)
    Restituisce (frame uint8, L_alta usata).
    """
    pix = arr.astype(np.float32)

    # --- selezione dei pixel ---
    dist = np.linalg.norm(pix - np.array(par['origine'], dtype=np.float32), axis=2)
    soglia = par['tol_or'] / 100 * DIST_MAX
    sfuma = max(par['sfuma'] / 100 * DIST_MAX, 1e-6)
    peso = (np.clip(1 - (dist - soglia) / sfuma, 0, 1) * zona)[..., None]

    # --- cambio di tonalità che conserva luci e ombre ---
    nuovo = np.array(par['nuovo'], dtype=np.float32)
    L = _lum(pix)
    Ls = float(_lum(np.array(par['origine'], dtype=np.float32)))
    Lt = float(_lum(nuovo))

    # punto "bianco" dei capelli: 95° percentile dei pixel selezionati
    if L_alta is None:
        sel = L[peso[..., 0] > 0.5]
        L_alta = float(np.percentile(sel, 95)) if sel.size > 50 else 0.3
        L_alta = max(L_alta, 0.15)

    n = L / L_alta

    # sollevamento dei neri, proporzionale a quanto il nuovo colore è più chiaro
    fattore = par['neri'] / 100
    lift = float(np.clip(max(Lt - Ls, 0) * 1.0 * fattore, 0, 0.8))
    n = lift + (1 - lift) * np.clip(n, 0, 1.5)

    # ombre -> colore scuro, medi -> colore pieno, riflessi -> verso il bianco
    base = nuovo * np.minimum(n, 1)[..., None]
    riflesso = np.clip(n - 1, 0, 0.5)[..., None] * 1.2
    colorato = base * (1 - riflesso) + 255.0 * riflesso

    colorato = np.clip(colorato, 0, 255)
    out = pix * (1 - peso) + colorato * peso
    return out.astype(np.uint8), L_alta


def mostra_mod(index):
    """Anteprima: frame `index` ridimensionato a 400 px, con il colore cambiato."""
    global frame_originale
    if video is None or in_render:
        return

    t = _tempo(video, index)
    img = Image.fromarray(video.get_frame(t))

    w, h = img.size
    rw, rh = 400, 400
    if w >= h:
        rh = (rw * h) // w
    else:
        rw = (rh * w) // h
    img = img.resize((rw, rh), Image.Resampling.BICUBIC)
    frame_originale = img

    par = leggi_parametri()
    # i px dello slider sono a risoluzione piena: li scalo per l'anteprima
    zona = _zona(t, img.size, par['bordi'] * img.size[0] / w)
    out, _ = elabora(np.asarray(img), zona, par)

    photo = ImageTk.PhotoImage(Image.fromarray(out))
    frame.delete('all')
    frame.create_image(200, 200, image=photo)
    frame.image = photo


def aggiorna(*_):
    """Ridisegna il frame corrente (usato dagli slider)."""
    if video is None or in_render:
        return
    mostra_mod(int(scorriframmi.get()))


# --- Scelta colori -----------------------------------------------------------
def aggiorna_colore_origine(rgb):
    global COLORE_ORIGINE
    COLORE_ORIGINE = tuple(int(c) for c in rgb)
    color_origine.config(bg=_hex(COLORE_ORIGINE))
    lab_color_origine.config(text=f'colore origine ({_hex(COLORE_ORIGINE)})')
    aggiorna()


def aggiorna_colore_nuovo(rgb):
    global COLORE_NUOVO
    COLORE_NUOVO = tuple(int(c) for c in rgb)
    color_cambiato.config(bg=_hex(COLORE_NUOVO))
    lab_color_cambiato.config(text=f'cambia colore in ({_hex(COLORE_NUOVO)})')
    aggiorna()


def palette_origine(*_):
    if in_render:
        return
    rgb, _h = colorchooser.askcolor(color=_hex(COLORE_ORIGINE),
                                    title='Colore origine', parent=window)
    if rgb:
        aggiorna_colore_origine(rgb)


def palette_nuovo(*_):
    if in_render:
        return
    rgb, _h = colorchooser.askcolor(color=_hex(COLORE_NUOVO),
                                    title='Nuovo colore', parent=window)
    if rgb:
        aggiorna_colore_nuovo(rgb)


# --- Contagocce --------------------------------------------------------------
def attiva_contagocce():
    global contagocce_attivo
    if frame_originale is None or in_render:
        return
    contagocce_attivo = True
    frame.config(cursor='crosshair')
    btn_contagocce.config(relief='sunken')


def disattiva_contagocce():
    global contagocce_attivo
    contagocce_attivo = False
    frame.config(cursor='')
    btn_contagocce.config(relief='raised')


def clic_frame(event):
    if not contagocce_attivo or frame_originale is None:
        return
    w, h = frame_originale.size
    x = event.x - (200 - w // 2)
    y = event.y - (200 - h // 2)
    if 0 <= x < w and 0 <= y < h:
        aggiorna_colore_origine(frame_originale.getpixel((x, y))[:3])
        disattiva_contagocce()


# --- Caricamento -------------------------------------------------------------
def f_load_v():
    global path_video, video
    file = filedialog.askopenfilename(
        filetypes=[('Video', '*.mp4 *.avi *.mov *.mkv'), ('Tutti i file', '*.*')]
    )
    if not file:
        return
    if video is not None:
        video.close()
    path_video = file
    video = VideoFileClip(file)

    n_frames = int(video.duration * video.fps)
    scorriframmi.config(to=max(n_frames - 1, 1))
    scorriframmi.set(1)
    mostra_mod(0)


def f_load_m():
    global path_mask, video_mask
    file = filedialog.askopenfilename(
        filetypes=[('Video', '*.mp4 *.avi *.mov *.mkv'), ('Tutti i file', '*.*')]
    )
    if not file:
        return
    if video_mask is not None:
        video_mask.close()
    path_mask = file
    video_mask = VideoFileClip(file)
    aggiorna()


def f_scorri(valore):
    if video is None or in_render:
        return
    mostra_mod(int(float(valore)))


# --- Rendering ---------------------------------------------------------------
def _imposta_controlli(attivi):
    stato = 'normal' if attivi else 'disabled'
    for w in (carica_video, carica_mascherino, btn_palette_o, btn_contagocce,
              btn_palette_n, tolleranza_or, tolleranza_camb, tolleranza_neri,
              scorriframmi, slider_bordi, btn_render):
        w.config(state=stato)
    btn_annulla.config(state='disabled' if attivi else 'normal')


def f_annulla():
    global annulla_render
    annulla_render = True


def f_render():
    global in_render, annulla_render
    if video is None:
        messagebox.showinfo('Render', 'Carica prima un video.')
        return

    out_path = filedialog.asksaveasfilename(
        defaultextension='.mp4',
        filetypes=[('MP4', '*.mp4')],
        initialfile=os.path.splitext(os.path.basename(path_video))[0] + '_colore.mp4'
    )
    if not out_path:
        return

    par = leggi_parametri()                      # parametri bloccati per tutto il render

    # luce massima calcolata una sola volta (frame corrente, risoluzione piena)
    t0 = _tempo(video, int(scorriframmi.get()))
    arr0 = video.get_frame(t0)
    _, L_fissa = elabora(arr0, _zona(t0, (arr0.shape[1], arr0.shape[0]), par['bordi']), par)

    durata = video.duration
    contatore = {'n': 0}

    def make_frame(t):
        if annulla_render:
            raise InterruptedError
        arr = video.get_frame(t)
        zona = _zona(t, (arr.shape[1], arr.shape[0]), par['bordi'])
        out, _ = elabora(arr, zona, par, L_fissa)
        contatore['n'] += 1
        perc = min(t / durata * 100, 100)
        barra['value'] = perc
        lab_prog.config(text=f'Rendering: {perc:.0f}%  (frame {contatore["n"]})')
        window.update()                          # tiene viva la finestra (e il tasto Annulla)
        return out

    in_render = True
    annulla_render = False
    _imposta_controlli(False)
    barra['value'] = 0
    completato = False

    try:
        try:
            nuovo_clip = VideoClip(frame_function=make_frame, duration=durata)
        except TypeError:                        # moviepy 1.x
            nuovo_clip = VideoClip(make_frame=make_frame, duration=durata)

        if video.audio is not None:
            if hasattr(nuovo_clip, 'with_audio'):
                nuovo_clip = nuovo_clip.with_audio(video.audio)
            else:
                nuovo_clip = nuovo_clip.set_audio(video.audio)

        nuovo_clip.write_videofile(out_path, fps=video.fps, codec='libx264',
                                   audio_codec='aac', logger=None)
        completato = True
    except InterruptedError:
        lab_prog.config(text='Render annullato')
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
    except Exception as e:
        lab_prog.config(text='Errore nel render')
        messagebox.showerror('Errore render', str(e))
    finally:
        in_render = False
        _imposta_controlli(True)
        aggiorna()

    if completato:
        barra['value'] = 100
        lab_prog.config(text='Render completato')
        messagebox.showinfo('Render', f'Video salvato in:\n{out_path}')


# --- Pulsanti ---
frame_button = tk.Frame(window, bg='gray')
frame_button.grid(row=0, column=0, sticky='nw')

carica_video = tk.Button(frame_button, text='Load Video', command=f_load_v)
carica_video.grid(row=0, column=0, sticky='nw', padx=10, pady=5)

carica_mascherino = tk.Button(frame_button, text='Load Mascherino', command=f_load_m)
carica_mascherino.grid(row=0, column=1, sticky='nw', padx=10, pady=5)

btn_render = tk.Button(frame_button, text='Render video', command=f_render)
btn_render.grid(row=0, column=2, sticky='nw', padx=10, pady=5)

btn_annulla = tk.Button(frame_button, text='Annulla render', command=f_annulla,
                        state='disabled')
btn_annulla.grid(row=0, column=3, sticky='nw', padx=10, pady=5)

# --- Canvas video ---
frame_canvas = tk.Frame(window, bg='gray')
frame_canvas.grid(row=1, column=0)

frame = tk.Canvas(frame_canvas, width=400, height=400, bg='red')
frame.grid(row=0, column=0)
frame.bind('<Button-1>', clic_frame)

# --- Colori e tolleranze ---
frame_canvas2 = tk.Frame(window, bg='gray')
frame_canvas2.grid(row=2, column=0, sticky='nw', padx=10, pady=5)

lab_color_origine = tk.Label(frame_canvas2, text=f'colore origine ({_hex(COLORE_ORIGINE)})')
lab_color_origine.grid(row=0, column=0, sticky='nw', padx=10, pady=5)

frame_origine = tk.Frame(frame_canvas2, bg='gray')
frame_origine.grid(row=1, column=0, sticky='nw', padx=10, pady=5)

color_origine = tk.Canvas(frame_origine, width=20, height=20,
                          bg=_hex(COLORE_ORIGINE), cursor='hand2')
color_origine.grid(row=0, column=0, padx=(0, 5))
color_origine.bind('<Button-1>', palette_origine)

btn_palette_o = tk.Button(frame_origine, text='Palette', command=palette_origine)
btn_palette_o.grid(row=0, column=1, padx=2)
btn_contagocce = tk.Button(frame_origine, text='Contagocce', command=attiva_contagocce)
btn_contagocce.grid(row=0, column=2, padx=2)

lab_color_cambiato = tk.Label(frame_canvas2, text=f'cambia colore in ({_hex(COLORE_NUOVO)})')
lab_color_cambiato.grid(row=0, column=1, sticky='nw', padx=10, pady=5)

frame_nuovo = tk.Frame(frame_canvas2, bg='gray')
frame_nuovo.grid(row=1, column=1, sticky='nw', padx=10, pady=5)

color_cambiato = tk.Canvas(frame_nuovo, width=20, height=20,
                           bg=_hex(COLORE_NUOVO), cursor='hand2')
color_cambiato.grid(row=0, column=0, padx=(0, 5))
color_cambiato.bind('<Button-1>', palette_nuovo)

btn_palette_n = tk.Button(frame_nuovo, text='Palette', command=palette_nuovo)
btn_palette_n.grid(row=0, column=1, padx=2)

tolleranza_or = tk.Scale(frame_canvas2, from_=0.10, to=100.0, resolution=0.10,
                         orient='horizontal', length=200, label='Tolleranza origine',
                         command=aggiorna)
tolleranza_or.set(20)
tolleranza_or.grid(row=2, column=0, sticky='nw', padx=10, pady=5)

tolleranza_camb = tk.Scale(frame_canvas2, from_=0.10, to=100.0, resolution=0.10,
                           orient='horizontal', length=200,
                           label='Sfumatura (tolleranza cambiato)',
                           command=aggiorna)
tolleranza_camb.set(10)
tolleranza_camb.grid(row=2, column=1, sticky='nw', padx=10, pady=5)

tolleranza_neri = tk.Scale(frame_canvas2, from_=0, to=200, resolution=1,
                           orient='horizontal', length=200,
                           label='Schiarisci neri %', command=aggiorna)
tolleranza_neri.set(100)
tolleranza_neri.grid(row=3, column=1, sticky='nw', padx=10, pady=5)

scorriframmi = tk.Scale(frame_canvas2, from_=1, to=100, resolution=1,
                        orient='horizontal', length=200, label='scorri frames',
                        command=f_scorri)
scorriframmi.grid(row=3, column=0, sticky='nw', padx=10, pady=5)

slider_bordi = tk.Scale(frame_canvas2, from_=-50, to=50, resolution=1,
                        orient='horizontal', length=200,
                        label='Bordi mascherino (px)  - riduci / + espandi',
                        command=aggiorna)
slider_bordi.set(0)
slider_bordi.grid(row=4, column=0, sticky='nw', padx=10, pady=5)

# --- Avanzamento render ---
lab_prog = tk.Label(frame_canvas2, text='Pronto')
lab_prog.grid(row=5, column=0, columnspan=2, sticky='nw', padx=10, pady=(10, 0))
barra = ttk.Progressbar(frame_canvas2, orient='horizontal', length=420,
                        mode='determinate', maximum=100)
barra.grid(row=6, column=0, columnspan=2, sticky='nw', padx=10, pady=5)

window.mainloop()