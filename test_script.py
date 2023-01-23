import torch
import torchvision
import torchvision.transforms as transforms
from skimage import io
import sys

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Недостаточно аргументов')
        exit()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = torchvision.models.efficientnet_b0(pretrained=True)

    path_to_model = sys.argv[2]

    model_load = torch.jit.load(path_to_model, map_location=torch.device(device))

    img_path = sys.argv[1]

    img = io.imread(img_path)

    my_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(200, 200)),
    ])

    img = my_transform(img.copy())

    model.eval()

    img = img.unsqueeze(0)

    scores = model(img)
    _, predictions = scores.max(1)

    print('Model says it\'s ', 'spoof' if predictions[0] == 1 else 'live', 'image')
