from tqdm.auto import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from ..utils.metrics import AverageMeter, compute_metrics, get_scale_from_dataset
from ..trainer_utils import EvalPrediction


class EvalMetrics:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def evaluate(self, model: nn.Module, dataset: Dataset, scale: int = None):
        if scale is None:
            if len(dataset) > 0:
                scale = get_scale_from_dataset(dataset)
            else:
                raise ValueError(f"Unable to calculate scale from empty dataset.")

        eval_dataloader = DataLoader(dataset=dataset, batch_size=1)
        epoch_psnr = AverageMeter()
        epoch_ssim = AverageMeter()
        for i, data in tqdm(enumerate(eval_dataloader), total=len(dataset), desc='Evaluating dataset'):
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            model.to(self.device)
            with torch.no_grad():
                preds = model(inputs)

            metrics = compute_metrics(EvalPrediction(predictions=preds, labels=labels), scale=scale)

            epoch_psnr.update(metrics['psnr'], len(inputs))
            epoch_ssim.update(metrics['ssim'], len(inputs))
        print(f'scale:{str(scale)}      eval psnr: {epoch_psnr.avg:.2f}     ssim: {epoch_ssim.avg:.4f}')
