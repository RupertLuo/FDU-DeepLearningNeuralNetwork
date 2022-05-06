echo "baseline"
python mid-homework/task1_train.py DATA.augmentation False DATA.cutout False DATA.cutmix False DATA.mixup False
echo "cutout"
python mid-homework/task1_train.py DATA.augmentation False DATA.cutout True DATA.cutmix False DATA.mixup False
echo "cutmix"
python mid-homework/task1_train.py DATA.augmentation False DATA.cutout False DATA.cutmix True DATA.mixup False
echo "mixup"
python mid-homework/task1_train.py DATA.augmentation False DATA.cutout False DATA.cutmix False DATA.mixup True