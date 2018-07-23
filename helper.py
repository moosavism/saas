class CIFAR10(torchvision.datasets.CIFAR10):
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

from torch.utils.data import DataLoader

def train_w(net, optimizer, criterion, trainloader, stop_nb=10**7):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        outputs = logsoft(outputs)

        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        confs, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    return 100.*correct/total, train_loss/(batch_idx+1)

def noaug_cifar10():             
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform_train, transform_test


def get_loaders_cifar(nb_labelled, dataset, batch_size):
        
    transform_train, transform_test = noaug_cifar10()
    trainset_l = CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    trainset_l.train_data = np.array(trainset_l.train_data) 
    trainset_l.train_labels = np.array(trainset_l.train_labels)       

    num_noise_label = int(nb_labelled*noise_rat)
    print num_noise_label
    print trainset_l.train_labels[0:num_noise_label]
    trainset_l.train_labels[0:num_noise_label] = np.random.randint(nb_class, size=num_noise_label)
    print trainset_l.train_labels[0:num_noise_label]
    
    print (trainset_l.train_data.shape, len(trainset_l.train_labels))
    trainloader_l = DataLoader(trainset_l, batch_size=batch_size, shuffle=True, num_workers=1)
    
    return trainloader_l

def test(net, criterion, trainloader_l):
    net.eval()
   
    correct = 0
    total = 0
    test_loss= 0
    for batch_idx, (inputs, targets) in enumerate(trainloader_l):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        outputs = logsoft(outputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    acc = 100.*correct/total

    return acc, test_loss/(batch_idx+1)