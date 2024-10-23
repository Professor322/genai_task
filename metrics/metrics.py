from utils.class_registry import ClassRegistry
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datasets.datasets import datasets_registry


metrics_registry = ClassRegistry()


@metrics_registry.add_to_registry(name="fid")
class FID:
    """
    https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html
    """

    def __init__(
        self, batch_size, features=64, reset_real_features=False, device="cpu"
    ):
        self.device = device
        self.fid = FrechetInceptionDistance(
            feature=features, reset_real_features=reset_real_features
        ).to(self.device)
        self.batch_size = batch_size
        self.real_data_processed = False

    def __call__(self, orig_path, synt_path):
        self.real_dataset = datasets_registry["fid_dataset"](orig_path, train=False)
        self.fake_dataset = datasets_registry["fid_dataset"](synt_path, train=False)

        self.real_dataloader = DataLoader(
            self.real_dataset, shuffle=False, batch_size=self.batch_size
        )
        self.fake_dataloader = DataLoader(
            self.fake_dataset, shuffle=False, batch_size=self.batch_size
        )
        # although this keeps real data cached
        self.fid.reset()
        # real data needs to be processed only once
        if not self.real_data_processed:
            # process real data
            for batch in tqdm(self.real_dataloader):
                batch = batch.to(self.device)
                self.fid.update(batch, real=True)
            self.real_data_processed = True

        # process fake data
        for batch in tqdm(self.fake_dataloader):
            batch = batch.to(self.device)
            self.fid.update(batch, real=False)

        return {"fid": self.fid.compute().item()}
