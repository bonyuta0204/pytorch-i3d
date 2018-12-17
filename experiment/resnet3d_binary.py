import torch
from pytorch_i3d import InceptionI3d
from torchvision import transforms
import videotransforms
from mit_data import MITDataset, make_label_binarizer

INDEX_FILE = "experiment/binary_class/binary_class.csv"
SPLIT_FILE = "experiment/binary_class/split.csv"

batch_size = 1
train_transforms = transforms.Compose([
    videotransforms.RandomCrop(225),
    videotransforms.RandomHorizontalFlip(),
])
test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

dataset = MITDataset(
    mode="train",
    transforms=train_transforms,
    index_file=INDEX_FILE,
    split_file=SPLIT_FILE)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=10,
    pin_memory=True)

val_dataset = MITDataset(
    mode="val",
    transforms=test_transforms,
    index_file=INDEX_FILE,
    split_file=SPLIT_FILE)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=10,
    pin_memory=True)

mlb = dataset.mlb
num_classes = len(dataset.mlb.classes_)

resnet = torchvision.models.resnet50(pretrained=True)
resnet.fc = nn.Linear(2048, num_classes)
model = I3ResNet(copy.deepcopy(resnet), NUM_FRAMES)
