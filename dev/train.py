import argparse
import torch
import torch.optim as optim
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision import datasets
from torch.optim.lr_scheduler import StepLR



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
      print(batch_idx, (data, target))


# TODO train_loader/test_loader in main()

def main(data_set):

    # Training Parameters
    parser = argparse.ArgumentParser(description='ViT Trainer')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test_batch_size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')

    args = parser.parse_args()

    torch.manual_seed(1)

    use_cuda = torch.cuda.is_available() #set runtime to GPU for 'True'
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = Compose([Resize((224, 224)), ToTensor()])

    train_dataset = data_set
    test_dataset = 0

    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    #test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = ViT().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        #test(model, device, test_loader)
        scheduler.step()

    torch.save(model.state_dict(), "trained_ViT.pt")


if __name__='__main__':
    main()
