from yacs.config import CfgNode as CN
_CN = CN()

_CN.name = ''
_CN.suffix =''
_CN.gamma = 0.85 #controlla l'effetto della correzione gamma sull'immagine di output del flusso ottico
'''
La correzione gamma è una trasformazione non lineare applicata alle immagini al fine di regolare il rapporto tra la luminosità
 dell'immagine in ingresso e quella visualizzata. In generale, può essere utilizzata per modificare il contrasto dell'immagine 
 e migliorare la resa visiva. Aumentando il valore di gamma, si potrebbe aumentare il contrasto del flusso ottico.
'''
_CN.max_flow = 400      # massimo valore assumibile dal flusso ottico
_CN.batch_size = 8      
_CN.sum_freq = 100      # frequenza con cui vien eseguita la somma dei gradienti
_CN.val_freq = 100000000    # frequenza con cui viene eseguita la validazione  del modello durante il training
_CN.image_size = [500, 500] # Da cambiare, nel nostro caso in 504x504 (dopo padding)
_CN.add_noise = False       
_CN.use_smoothl1 = False    # se utilizzare funzione di errore oppure no
_CN.critical_params = []

_CN.network = 'MOFNetStack'

_CN.model = 'VideoFlow_ckpt/MOF_sintel.pth'
_CN.input_frames = 5

_CN.restore_ckpt = None

################################################
################################################
_CN.MOFNetStack = CN()
_CN.MOFNetStack.pretrain = True     # Se usare modello preaddestrato o no
_CN.MOFNetStack.Tfusion = 'stack'   # Tipo di fusione temporale (come vengono unite info di più frame)
_CN.MOFNetStack.cnet = 'twins'      # Tipo di rete convoluzionale
_CN.MOFNetStack.fnet = 'twins'      # Tipo di rete di flusso
_CN.MOFNetStack.down_ratio = 8      # rapporto di downsampling (riduzione dim spaziale delle feature map)
_CN.MOFNetStack.feat_dim = 256      # dimensione feature utilizzate nell'architettura
_CN.MOFNetStack.corr_fn = 'default' # funzione di correlazione utilizzata
_CN.MOFNetStack.corr_levels = 4     # livelli di correlazione, ovvero le diverse scale di dettaglio 
_CN.MOFNetStack.mixed_precision = True
_CN.MOFNetStack.context_3D = False

_CN.MOFNetStack.decoder_depth = 32  # Responsabile della generazione del flusso dinale
_CN.MOFNetStack.critical_params = ["cnet", "fnet", "pretrain", 'corr_fn', "Tfusion", "corr_levels", "decoder_depth", "mixed_precision"]

### TRAINER
_CN.trainer = CN()
_CN.trainer.scheduler = 'OneCycleLR'
_CN.trainer.optimizer = 'adamw'
_CN.trainer.canonical_lr = 12.5e-5
_CN.trainer.adamw_decay = 1e-4
_CN.trainer.clip = 1.0
_CN.trainer.num_steps = 90000
_CN.trainer.epsilon = 1e-8
_CN.trainer.anneal_strategy = 'linear'

def get_cfg():
    return _CN.clone()
# consente di ottenere una nuova copia della configurazione ogni volta che viene chiamata, 
# evitando così modifiche accidentali alla configurazione principale
