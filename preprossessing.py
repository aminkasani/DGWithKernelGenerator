from torchvision.transforms import v2

transformTrain = v2.Compose([
    v2.ToImage(),  # Converts PIL or ndarray to tensor Image object
    # v2.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
    # v2.RandomCrop(224),
    # v2.RandomInvert(p=0.1), #v1.3
    v2.RandomHorizontalFlip(p=0.2),
    v2.RandomVerticalFlip(p=0.2),  # Not yet in v2, you'd need a custom transform
    v2.RandomAffine(degrees=0, scale=(0.8, 1.2), shear=0),
    v2.RandomApply([v2.RandomRotation(degrees=90)], p=0.3),
    v2.RandomApply([v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.2),
    v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))], p=0.05),
    # v2.RandomErasing(p=0.8, scale=(0.02, 0.2), ratio=(0.5, 2.0), value=0),  # Not yet in v2 Compose
    v2.ToDtype(torch.float32, scale=True),  # Normalizes to [0,1] like ToTensor()
    v2.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])
    

    
    
    transformTest = v2.Compose([
    # v2.Resize((224, 224)),
    v2.ToImage(),
    # v2.RandomInvert(p=1.0), #v1.3
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])
