import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

train_data = datasets.CIFAR10(root="./data", download=True, train=True)
test_data = datasets.CIFAR10(root="./data", train=False)


class CustomDataset(torch.utils.data.Dataset):
    """
    this is a custom test to concatenate train_data and
    images created during data augumentation
    """

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


def image_augumentation(dataset):
    """
    this generates 8 extra images for each images in train_data
    """
    transformed_images = []
    labels = []
    transform_list = [
        (transforms.ColorJitter(brightness=0.5, hue=0.3), 2),
        (transforms.RandomRotation(degrees=(0, 180)), 2),
        (transforms.RandomPosterize(bits=2), 1),
        (transforms.RandomAdjustSharpness(sharpness_factor=2), 1),
        (transforms.RandomHorizontalFlip(p=1), 1),
        (transforms.RandomVerticalFlip(p=1), 1),
    ]
    to_tensor = transforms.ToTensor()
    for i in range(len(train_data)):
        image, label = dataset[i]
        for transform, count in transform_list:
            for _ in range(count):
                transformed_images.append(to_tensor(transform(image)))
                labels.append(label)
        print("transormation done", i, "/", len(dataset))
    return transformed_images, labels


augumented_images = image_augumentation(train_data)

augument_dataset = CustomDataset(augumented_images)

train_data = torch.utils.data.ConcatDataset((train_data, augument_dataset))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=150, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=150)
