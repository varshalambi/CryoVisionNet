##### transforms.py #####
from monai.transforms import Compose, RandFlipd, RandRotate90d, RandCropByLabelClassesd

def get_train_transforms(num_samples=16):
    """
    Define the transformations to be applied to the training data.
    """
    return Compose([
        RandCropByLabelClassesd(keys=["image", "label"], label_key="label",
                                spatial_size=[96, 96, 96], num_classes=7, num_samples=num_samples),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 2]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    ])

