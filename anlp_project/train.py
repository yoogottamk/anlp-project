import pytorch_lightning as pl

from anlp_project.models.anlp_project import ANLPProject

class ANLPProjectModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = ANLPProject(config)

    def configure_optimizers(self):
        pass

def train_model(config):
    pass
