import segmentation_models_pytorch as smp
from segmentation_models_pytorch.metrics.functional import f1_score

IGNORE_INDEX = 255   # label value for no-data pixels; excluded from loss and metrics (must match loader.py)
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
from datetime import datetime
from loader import GMKDataset
from torchvision import transforms
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt

class GMM:

    def __init__(self, params):
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = self.params["training"]["epochs"]
        self.batch_size = self.params["training"]["batch_size"]
        self.out_dir = self.params["data"]["out"]

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        else:
            raise FileExistsError("%s already exists." % (self.out_dir))

        print("....initializing model.")
        if self.params["model"]["name"] == "DeepLabV3Plus":
            model = smp.DeepLabV3Plus(
                encoder_name = self.params["model"]["encoder"],
                encoder_weights = self.params["model"]["weights"],
                in_channels = self.params["model"]["channels"],
                classes = self.params["model"]["classes"]
            )
        elif self.params["model"]["name"] == "Unet++":
            model = smp.UnetPlusPlus(
                encoder_name=self.params["model"]["encoder"],
                encoder_weights=self.params["model"]["weights"],
                in_channels=self.params["model"]["channels"],
                classes=self.params["model"]["classes"]
            )
        elif self.params["model"]["name"] == "Segformer":
            model = smp.Segformer(
                encoder_name=self.params["model"]["encoder"],
                encoder_weights=self.params["model"]["weights"],
                in_channels=self.params["model"]["channels"],
                classes=self.params["model"]["classes"]
            )
        elif self.params["model"]["name"] == "MAnet":
            model = smp.MAnet(
                encoder_name=self.params["model"]["encoder"],
                encoder_weights=self.params["model"]["weights"],
                in_channels=self.params["model"]["channels"],
                classes=self.params["model"]["classes"]
            )
        else:
            raise ValueError("%s not supported." % (self.params["model"]["name"]))

        # ── Optimizer ────────────────────────────────────────────────────────
        opt_type = self.params["training"]["optimizer"]["type"]
        lr       = self.params["training"]["optimizer"]["lr"]
        wd       = self.params["training"]["optimizer"].get("weight_decay", 0.0)

        if opt_type == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_type == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif opt_type == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError("%s not supported." % opt_type)

        # ── Loss ─────────────────────────────────────────────────────────────
        loss_name = self.params["training"]["loss"]

        if loss_name == "dice":
            loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True, ignore_index=IGNORE_INDEX)
        elif loss_name == "focal":
            loss_fn = smp.losses.FocalLoss(smp.losses.MULTICLASS_MODE, ignore_index=IGNORE_INDEX)
        elif loss_name == "lovasz":
            loss_fn = smp.losses.LovaszLoss(smp.losses.MULTICLASS_MODE, from_logits=True, ignore_index=IGNORE_INDEX)
        elif loss_name == "dice_ce":
            # Compound loss: 0.5 * Dice + 0.5 * SoftCE with label smoothing
            # More stable early training; CE provides pixel-level gradient signal
            # that pure Dice lacks, especially for rare classes.
            _dice = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True, smooth=1e-6, ignore_index=IGNORE_INDEX)
            _ce   = smp.losses.SoftCrossEntropyLoss(smooth_factor=0.1, reduction="mean", ignore_index=IGNORE_INDEX)
            loss_fn = lambda pred, target: 0.5 * _dice(pred, target) + 0.5 * _ce(pred, target)
        else:
            raise ValueError("%s not supported." % loss_name)

        # ── Preprocessing transform ───────────────────────────────────────────
        normalize = self.params["model"].get("normalize", True)  # default True to preserve original behaviour
        if self.params["model"]["weights"] is not None and normalize:
            preprocessing_params = smp.encoders.get_preprocessing_params(self.params["model"]["encoder"], self.params["model"]["weights"])
            pre_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=preprocessing_params["mean"],
                                     std=preprocessing_params["std"])
                ])
        else:
            # normalize=False: data is already in [0,1] — just convert to tensor
            pre_transform = transforms.Compose([
                transforms.ToTensor(),
                ])

        # ── Datasets & loaders ───────────────────────────────────────────────
        augment = self.params["training"].get("augment", False)
        print("...preparing training.%s" % (" Augmentation ON." if augment else ""))

        train_data = GMKDataset(self.params["data"]["train"],
                                device=self.device,
                                classes=self.params["model"]["classes"],
                                transform=pre_transform,
                                augment=augment)

        valid_data = GMKDataset(self.params["data"]["valid"],
                                device=self.device,
                                classes=self.params["model"]["classes"],
                                transform=pre_transform,
                                augment=False)   # never augment validation

        self.train_loader = DataLoader(train_data, batch_size=self.params["training"]["batch_size"], shuffle=True, drop_last=True)
        self.valid_loader = DataLoader(valid_data, batch_size=self.params["training"]["batch_size"], shuffle=False)

        # ── Scheduler ─────────────────────────────────────────────────────────
        sched_cfg = self.params["training"]["scheduler"]
        sched_type = sched_cfg.get("type", "multistep") if sched_cfg else "multistep"

        if sched_cfg is None:
            # dummy scheduler that never changes LR
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=1.0)
        elif sched_type == "cosine":
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.epochs,
                eta_min=sched_cfg.get("eta_min", 1e-6)
            )
        else:
            # original multistep behaviour — fully backwards compatible
            scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=sched_cfg["steps"],
                                                 gamma=sched_cfg["factor"])

        model.to(self.device)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn

    def train_loop(self):

        self.log_path = os.path.join(self.out_dir, "train.log")
        n_classes = self.params["model"]["classes"]

        log_file = open(self.log_path, "w")

        # Header: micro metrics + per-class F1
        per_class_f1_header = ";".join([f"f1_class_{c+1}" for c in range(n_classes)])
        log_file.write(f"epoch;time;train_loss;valid_loss;valid_iou_micro;valid_f1_micro;{per_class_f1_header}\n")

        chkpnts_dir = os.path.join(self.out_dir, "chkpnts")
        os.makedirs(chkpnts_dir, exist_ok=True)

        with open(os.path.join(self.out_dir, 'config.yaml'), 'w') as yaml_file:
            yaml.dump(self.params, yaml_file)

        # ── One-time validation class distribution ────────────────────────────
        print("...scanning validation labels.")
        val_pixels_pc = torch.zeros(n_classes, dtype=torch.long)
        with torch.no_grad():
            for batch in self.valid_loader:
                labels = batch[1].view(-1)   # flatten all spatial positions
                for c in range(n_classes):
                    val_pixels_pc[c] += (labels == c).sum()
        total_valid_px = val_pixels_pc.sum().item()
        print("  Validation class distribution (no-data excluded):")
        print("  " + "  ".join([
            "c%02d: %d px (%.1f%%)" % (c+1, val_pixels_pc[c].item(), 100*val_pixels_pc[c].item()/max(total_valid_px, 1))
            for c in range(n_classes)
        ]))

        for epoch in range(self.epochs):
            print('EPOCH {}:'.format(epoch+1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            train_loss = 0

            for batch in tqdm(self.train_loader, desc="Training"):

                # Every batch instance is an input + label pair
                inputs = batch[0].to(self.device)
                labels = torch.squeeze(batch[1]).to(self.device)

                # Zero your gradients for every batch!
                self.optimizer.zero_grad()

                # Make predictions for this batch
                outputs = torch.squeeze(self.model(inputs))

                # Compute the loss and its gradients
                loss = self.loss_fn(outputs, labels)
                loss.backward()

                # Adjust learning weights
                self.optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(self.train_loader)

            self.scheduler.step()

            # Disable gradient computation and reduce memory consumption.
            self.model.eval()
            val_los = 0

            # Accumulators for micro metrics
            tp, fp, fn, tn = 0, 0, 0, 0

            # Accumulators for per-class metrics (shape: n_classes)
            tp_pc = torch.zeros(n_classes)
            fp_pc = torch.zeros(n_classes)
            fn_pc = torch.zeros(n_classes)
            tn_pc = torch.zeros(n_classes)

            with torch.no_grad():
                for batch in tqdm(self.valid_loader, desc="Evaluate"):
                    val_inputs = batch[0].to(self.device)
                    val_labels_true = batch[1].to(self.device)

                    val_outputs = self.model(val_inputs)
                    assert (val_outputs.shape[1] == n_classes)  # [batch_size, number_of_classes, H, W]

                    #loss_fn requires "raw" model outputs; accordingly, we apply the log_softmax afterwards
                    curr_val_loss = self.loss_fn(val_outputs, val_labels_true)
                    val_los += curr_val_loss

                    val_probs = val_outputs.log_softmax(dim=1).exp()

                    val_labels_pred = torch.argmax(val_probs, dim=1, keepdim=True)
                    val_labels_pred = val_labels_pred.long()

                    batch_tp, batch_fp, batch_fn, batch_tn = smp.metrics.get_stats(
                        val_labels_pred, val_labels_true,
                        mode='multiclass', num_classes=n_classes,
                        ignore_index=IGNORE_INDEX
                    )

                    # micro accumulators (scalar)
                    tp += batch_tp.sum().item()
                    fp += batch_fp.sum().item()
                    fn += batch_fn.sum().item()
                    tn += batch_tn.sum().item()

                    # per-class accumulators (sum over batch dim, keep class dim)
                    tp_pc += batch_tp.sum(dim=0).cpu()
                    fp_pc += batch_fp.sum(dim=0).cpu()
                    fn_pc += batch_fn.sum(dim=0).cpu()
                    tn_pc += batch_tn.sum(dim=0).cpu()

            # ── Micro metrics ────────────────────────────────────────────────
            epoch_iou_score = smp.metrics.iou_score(
                torch.tensor([tp]), torch.tensor([fp]),
                torch.tensor([fn]), torch.tensor([tn]),
                reduction="micro").item()

            epoch_f1_score = smp.metrics.f1_score(
                torch.tensor([tp]), torch.tensor([fp]),
                torch.tensor([fn]), torch.tensor([tn]),
                reduction="micro").item()

            # ── Per-class F1 ─────────────────────────────────────────────────
            per_class_f1 = smp.metrics.f1_score(
                tp_pc, fp_pc, fn_pc, tn_pc,
                reduction="none"           # returns tensor of shape (n_classes,)
            ).numpy()

            avg_val_loss = val_los / len(self.valid_loader)

            print('LOSS train: %.4f valid: %.4f - IOU: %.2f - F1: %.2f' % (
                avg_train_loss, avg_val_loss, epoch_iou_score*100, epoch_f1_score*100))
            print('  Per-class F1: ' + '  '.join([
                'c%02d:%.1f' % (c+1, per_class_f1[c]*100) for c in range(n_classes)
            ]))

            per_class_f1_str = ";".join(["%.2f" % (v*100) for v in per_class_f1])
            log_file.write("%i;%s;%.4f;%.4f;%.2f;%.2f;%s\n" % (
                epoch+1,
                datetime.now().strftime('%Y%m%d_%H%M%S'),
                avg_train_loss, avg_val_loss,
                epoch_iou_score*100, epoch_f1_score*100,
                per_class_f1_str
            ))
            log_file.flush()
            os.fsync(log_file)

            # Track best performance, and save the model's state
            model_path = os.path.join(chkpnts_dir, 'epoch_{}'.format(epoch+1))
            self.model.save_pretrained(model_path)

    def plot_train_val_loss(self):

        out_path = os.path.join(os.path.dirname(self.log_path), "train_val_loss.png")
        pd_train_log = pd.read_csv(self.log_path, sep=";")

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
        ax1.plot(pd_train_log.epoch, pd_train_log.train_loss, label="Training")
        ax1.plot(pd_train_log.epoch, pd_train_log.valid_loss, label="Validation")
        ax1.set_title("Loss")
        ax1.legend()
        ax1.grid(True, linestyle="--")

        ax2.plot(pd_train_log.epoch, pd_train_log.valid_iou_micro, label="IOU")
        ax2.plot(pd_train_log.epoch, pd_train_log.valid_f1_micro, label="F1")
        ax2.set_title("Quality")
        ax2.legend()
        ax2.grid(True, linestyle="--")

        plt.savefig(out_path, dpi=300, bbox_inches="tight")
