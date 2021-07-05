#libraries

def train((args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
      pass


# TODO train_loader/test_loader in main()

def main():

    torch.manual_seed(args.seed)

    use_cuda = torch.cuda.is_available() #set runtime to GPU for 'True'
    device = torch.device("cuda" if use_cuda else "cpu")

    transform = Compose([Resize((224, 224)), ToTensor()])

    train_dataset = 0
    test_dataset = 0

    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = ViT().to(device)
