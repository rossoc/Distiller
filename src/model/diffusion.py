import lightning as L
from .encoder import gemma_encoder
from .decoder import Decoder


class Diffusion(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.encoder = gemma_encoder()
        self.decoder = Decoder
