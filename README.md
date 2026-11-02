editing di computer vision cambia faccia e cambia colore capelli.
clona report: git clone: https://github.com/asprho-arkimete/visualkey.git
cd visualkey
crea ambiente virtuale:python 3.10 -m venv vvisual
attiva ambiente: cd vvisual/Scripts , activate, cd.. cd..
istalla requisiti: pip install -r requisiti.txt
scarica file yolo face model: https://github.com/akanametov/yolo-face nella dir principale,
clona e istalla: https://github.com/based9based/based-roop

una volta istallato tutto prosegui:
1] usa lo script python select_actor.py per separare i differenti personaggi tramite riconoscimento facciale ,
in differenti video con trasparenza, un video ogni personagio;
*se non riesci in automatico usa un software come after effects, con delle maschere*. 
2] usa il repo roto per cambiare le facce nei video:
python run.py --target "video_dest.mp4" --source "image_sorgente.jpg" -o swapped.mp4 --execution-provider cuda --frame-processor face_swapper face_enhancer --keep-audio --keep-fps --execution-threads 24
con 16GB VRAM lascia -threads 24 ,altrimenti aumenta o diminuisci in base alla tua GPU

3] apri lo script: visualkey.py
trascina il video originale o modificati e seleziona rileva capelli , creera dei video mascherino dei capelli dei soggetti.

4] apri: cambia_colore.py
seleziona il video modificato con le facce e il mascherino dei capelli, cambia il colore dei capelli a piace e renderizza
5 unisci i 2 video con editing video (after effects) e usa la traccia originale della seleziona personaggi che ha il canale trasparenza come mascherino 
dei video modificati: (facce e capelli), in fine renderizza tutto

Nota i visi di  profilo laterale non sono molto accurati con la sostituzione dei volti con based9based
based-roop, se si conosce usare un sistema migliore.









