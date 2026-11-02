# VisualKey

Editing video con la computer vision: **cambia faccia** e **cambia colore dei capelli**, con interfaccia grafica (tkinter) e accelerazione GPU NVIDIA.

## Cosa fa

| Strumento | Descrizione |
|-----------|-------------|
| **Select Actor** | Rileva i volti nel video (YOLO face + insightface), li raggruppa per persona ed esporta una clip **ProRes 4444 con canale alpha** per ogni persona selezionata. Permette di limitare manualmente il numero di persone presenti nella scena. |
| **Cambio colore capelli** | Cambia il colore dei capelli in un video usando un **video mascherino** bianco/nero. Uno slider permette di espandere o ridurre i bordi della zona bianca del mascherino. |
| **Cambia faccia** | Sostituzione del volto nel video. |

## Requisiti

- Windows
- GPU NVIDIA con CUDA (testato su RTX 4060 16 GB)
- Python 3.10
- [FFmpeg](https://ffmpeg.org/) (incluso tramite `imageio-ffmpeg`)

Librerie principali: `torch`, `ultralytics`, `insightface`, `onnxruntime-gpu`, `opencv-python`, `moviepy`, `Pillow`, `tkinterdnd2`, `sam3`.

## Installazione

```bash
git clone https://github.com/asprho-arkimete/visualkey.git
cd visualkey

conda create -n vkey python=3.10
conda activate vkey

pip install -r requisiti.txt
```

> `requisiti.txt` usa l'indice extra di PyTorch per le versioni CUDA (`torch==2.14.1+cu132`).

## Modelli

Al primo avvio vengono usati (o scaricati) i seguenti modelli:

- `yolov12l-face` per il rilevamento dei volti
- `buffalo_l` (insightface) per il riconoscimento e il confronto dei volti
- SAM 3 per la segmentazione

Posiziona i file dei modelli nella cartella indicata nello script, se non vengono scaricati automaticamente.

## Utilizzo

Avvia lo script desiderato:

```bash
python <nome_script>.py
```

### Select Actor

1. Trascina il video nella finestra (drag & drop).
2. Premi il pulsante di analisi: i volti vengono numerati come persone.
3. Se nel video sai che ci sono N attori, inserisci N nel campo del numero massimo di persone: il programma assegna ogni volto alla persona più somigliante.
4. Spunta le persone che vuoi esportare.
5. Esporta: ottieni un `.mov` ProRes 4444 con alpha per ogni persona.

### Cambio colore capelli

1. Carica il video originale e il video mascherino (capelli in bianco, resto in nero).
2. Regola con lo slider l'espansione o la riduzione dei bordi del mascherino.
3. Scegli il colore e avvia l'elaborazione.

## Struttura del repository

```
visualkey/
├── requisiti.txt
├── README.md
└── *.py        # script dei vari strumenti
```

## Note

- Il progetto è pensato per l'uso locale su GPU NVIDIA; senza CUDA i tempi di elaborazione aumentano molto.
- Le clip con alpha esportate in ProRes 4444 sono compatibili con i principali software di montaggio.









