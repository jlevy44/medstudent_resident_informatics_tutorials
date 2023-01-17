import torch
import glob
import flash
from flash.core.data.utils import download_data
from flash.image import SemanticSegmentation, SemanticSegmentationData
import matplotlib.pyplot as plt

datamodule = SemanticSegmentationData.from_folders(
    train_folder="train/imgs",
    train_target_folder="train/labels",
    val_split=0.1,
    transform_kwargs=dict(image_size=(256, 256)),
    num_classes=3,
    batch_size=4,
)

model = SemanticSegmentation(
    backbone="mobilenetv3_large_100",
    head="fpn",
    num_classes=datamodule.num_classes,
)

trainer = flash.Trainer(max_epochs=10, gpus=None)
trainer.finetune(model, datamodule=datamodule)#, strategy="freeze"

datamodule = SemanticSegmentationData.from_files(
    predict_files=sorted(glob.glob("test/imgs/*.png")),
    batch_size=3,
)
predictions = trainer.predict(model, datamodule=datamodule)

from functools import reduce
predictions_list=list(map(lambda x: torch.softmax(x['preds'],0).numpy(),reduce(lambda x,y:x+y,predictions)))

plt.imshow(predictions_list[7][1])