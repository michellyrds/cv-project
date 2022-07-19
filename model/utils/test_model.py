from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import max as tmax, exp as texp



def testModel(model, device):
    model.eval()
    data = datasets.ImageFolder('./media/input', transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
    loader = DataLoader(data, batch_size=1, shuffle=True)
    dataIter = iter(loader)

    

    for inputs, labels in dataIter:
        inputs = inputs.to(device)
        labels = labels.to(device)

        ps = texp(model.forward(inputs))
        _, predTest = tmax(ps,1) 



def save_max_score(scores):
    ...

