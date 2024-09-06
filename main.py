from PIL import Image
import os
import pytorch_lightning as L
import torch
# from model import SuperResolutionVAEModel, SuperResolutionVAEConfig
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from pytorch_lightning.callbacks import ModelCheckpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

class LitSuperResolutionVAE(L.LightningModule):
    def __init__(self):
        super().__init__()
        config = SuperResolutionVAEConfig()
        self.model = SuperResolutionVAEModel(config)
        self.psnr_metric = PeakSignalNoiseRatio()
        self.ssim_metric = StructuralSimilarityIndexMeasure()
    def forward(self, x, targets=None):
        return self.model(x, targets) # forward step -> meant for inference only



    def training_step(self, batch, _):
        x, target = batch
        # pytorch lightning automatically runs the backward pass and optimizer step
        #self.optimizer.zero_grad()
        y, loss = self(x, target)

        self.psnr_metric.update(y, target) # update the psnr metric for generated image and target image
        self.ssim_metric.update(y, target) # update the ssim metric for generated image and target image
        self.log('train_loss', loss)
        #loss.backward()
        #self.optimizer.step()
        #acc = accuracy(logits, label, task="multiclass", num_classes=10)
        #self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, _):
        x, target = batch
        y, loss = self(x, target)
        self.psnr_metric.update(y, target) # update the psnr metric for generated image and target image
        self.ssim_metric.update(y, target) # update the ssim metric for generated image and target image
        self.log('val_loss', loss)
        #acc = accuracy(logits, label, task="multiclass", num_classes=10)
        #self.log('train_acc', acc)
        return loss

    def test_step(self, batch, _):
        x, target = batch
        _, loss = self(x, target)
        self.log('test_loss', loss)
        return loss

    """
    def test_epoch_end(self, outs):
        avg_loss = torch.stack([x for x in outs]).mean()
        self.log("avg_test_loss", avg_loss)
        psnr = self.psnr_metric.compute()
        ssim = self.ssim_metric.compute()
        self.log("val_psnr", psnr)
        self.log("val_ssim", ssim)
    """

    def on_validation_epoch_end(self):
        psnr = self.psnr_metric.compute()
        ssim = self.ssim_metric.compute()
        self.log("val_psnr", psnr)
        self.log("val_ssim", torch.stack(ssim).numpy() if isinstance(ssim, tuple) else ssim)
        self.psnr_metric.reset()
        self.ssim_metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

# function for inference 
def upscale_image(model, lr_image):
    model.eval()
    with torch.no_grad():
        lr_image = lr_image.unsqueeze(0)  # Add batch dimension
        recon_image = model(lr_image)
        return recon_image.squeeze(0)  # Remove batch dimension


# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',   # Metric to monitor
    filename='checkpoint',  # Filename for the best checkpoint
    save_top_k=1,          # Save only the best model
    mode='min'             # Save the model with minimum validation loss
)


if __name__ == "__main__":
    import numpy as np
    from gpt_dataset import get_dataloader



    _type = input("train / inference, yes if training (y/N): ")
    train_dset, val_dset, test_dset = get_dataloader()
    if _type == "y":
        lit_model = LitSuperResolutionVAE()
        trainer = L.Trainer(max_epochs=2, callbacks=[checkpoint_callback])
        trainer.fit(lit_model, train_dataloaders=train_dset, val_dataloaders=val_dset)

        trainer.test(lit_model, test_dset)

    else:
        model = LitSuperResolutionVAE.load_from_checkpoint(f"checkpoint.ckpt")
        model.eval()

        # Perform inference with the loaded model
        
        
        
        # Upscale a sample image
        sample_idx = int(input(f"Enter an index between 0 and {len(test_dset) - 1}: "))
        img_name = input("Enter the name of the image: ")

        # check if image name exists
        if os.path.exists(img_name):
            raise ValueError("Image name already exists")

        lr_image, hr_image = test_dset.dataset[sample_idx]
        upscaled_image = upscale_image(model, lr_image)
        print(upscaled_image.shape)

        # Convert the upscaled image to a format that can be saved
        upscaled_image = upscaled_image.permute(1, 2, 0).cpu().numpy()  # Convert to HWC format and numpy array
        upscaled_image = (upscaled_image * 255).astype(np.uint8)  # Convert to uint8

        # Save the upscaled image to a file
        upscaled_image_pil = Image.fromarray(upscaled_image)
        upscaled_image_pil.save(os.path.join("generated", img_name))
        # Save original image
        orig_image = Image.fromarray(lr_image.permute(1, 2, 0).numpy())
        orig_image.save(os.path.join("generated", f"orig_{img_name}"))

        print(f"Upscaled image saved to 'generated/{img_name}'")
        #print(f"Total accuracy of predictions: {total_predictions * 100 / len(test_data):.3f}%")
