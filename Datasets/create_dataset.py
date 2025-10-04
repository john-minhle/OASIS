'''
Split dataset as train, test, val  6:2:2
use function dataset_wrap, return {train:, val:, test:} torch dataset

datasets names: isic2018, PH2, DMF, SKD
'''
import cv2
import os
import json
import torch
import random
import numpy as np
from torchvision import transforms
import albumentations as A
import pandas as pd
from Datasets.transform import *
from Datasets.unimatch_utils import obtain_cutmix_box
from PIL import Image, ImageEnhance
# from Datasets.data_augmentation import DataAugmentation
dataset_indices = {
    'CVC_clinicDB': 0,
}


def norm01(x):
    return np.clip(x, 0, 255) / 255


def process_multiclass_label(label_data):
    """Convert label image to one-hot encoded tensor"""
    num_classes = 3  
    h, w = label_data.shape
    one_hot = np.zeros((num_classes, h, w), dtype=np.float32)
    
    for i in range(num_classes):
        one_hot[i][label_data == i] = 1
        
    return one_hot

    
class StrongWeakAugment(torch.utils.data.Dataset):
    def __init__(self, dataset, img_size, use_aug=False, data_path='./proceeded_data/'):
        super(StrongWeakAugment, self).__init__()
        
        self.dataset = dataset
        self.root_dir = data_path
        self.use_aug = use_aug

        self.num_samples = len(self.dataset)

        w_p = 0.3
        s_p = 0.7
        self.weak_augment = A.Compose([
            A.Resize(img_size, img_size),
            A.ElasticTransform(
                alpha=10, 
                sigma=2,   
                p=w_p
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=w_p
            ),
            A.GaussianBlur(
                blur_limit=(3, 5),
                p=w_p
            ),
        ])
        self.strong_augment = A.Compose([
            A.GaussNoise(
                var_limit=(10.0, 50.0),
                p=s_p
            ),
            A.ElasticTransform(
                alpha=40, 
                sigma=4,   
                p=s_p
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=s_p
            ),
            A.GaussianBlur(
                blur_limit=(3, 7),
                p=s_p
            ),
            A.MotionBlur(
                blur_limit=(3, 7),
                p=s_p
            ),
        ])
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
    

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample_name = self.dataset[index]
        img_path = os.path.join(self.root_dir, f'images/{sample_name}')
        label_path = os.path.join(self.root_dir, f'labels/{sample_name}')

        img_data = np.load(img_path)

        img_w = self.weak_augment(image=img_data.astype('uint8'))['image']
        img_s = self.strong_augment(image=img_w.astype('uint8'))['image']
        
        img_w = norm01(img_w)
        img_s = norm01(img_s)
       
        img_w = torch.from_numpy(img_w).float()
        img_s = torch.from_numpy(img_s).float()

        img_w = img_w.permute(2, 0, 1)
        org_img = img_w
        img_s = img_s.permute(2, 0, 1)

        return{
            'id': index,
            'img_w': img_w,
            'img_s': img_s,
            'org_img': org_img,
        }

    def __len__(self):
        return self.num_samples

# def rotate_with_top_center_black_bg(image):
#     h, w = image.shape[:2]
#     center = (w // 2, 0)
#     angle = random.uniform(-10, 10)  # ngẫu nhiên từ -10 đến 10 độ

#     rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

#     rotated = cv2.warpAffine(image, rot_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
#     return rotated
# def rotate_with_top_center_black_bg_s(image):
#     h, w = image.shape[:2]
#     center = (w // 2, 0)
#     angle = random.uniform(-20, 20)  # ngẫu nhiên từ -20 đến 20 độ

#     rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

#     rotated = cv2.warpAffine(image, rot_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
#     return rotated
class RotateTopCenter:
    def __init__(self, angle_range=(-15, 15), p=0.2):
        self.angle_range = angle_range
        self.p = p

    def __call__(self, image, **kwargs):
        if np.random.rand() > self.p:
            return image

        angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
        h, w = image.shape[:2]
        center = (w // 2, 0)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        return rotated
class StrongWeakAugment4(torch.utils.data.Dataset):
    def __init__(self, dataset, img_size, use_aug=False, data_path='./proceeded_data/'):
        super(StrongWeakAugment4, self).__init__()
        
        self.dataset = dataset
        self.root_dir = data_path
        self.use_aug = use_aug
        self.num_samples = len(self.dataset)

        # p_w = 0.2  # Xác suất augment
        # p_s = 0.5
        # # Weak augmentation: thêm rotate ngẫu nhiên với nền đen
        # self.weak_augment = A.Compose([
        #     A.Resize(img_size, img_size),
        #     A.Lambda(image=RotateTopCenter(angle_range=(-15, 15), p=p_w)),
        #     A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=p_w),  # thay đổi ánh sáng nhẹ

        #     # A.ElasticTransform(alpha=10, sigma=2, p=p_w),
        # ])
        
        # # Strong augmentation
        # self.strong_augment = A.Compose([
        #     A.Resize(img_size, img_size),
        #     A.ElasticTransform(alpha=30, sigma=4, p=p_s),
        #     A.Lambda(image=RotateTopCenter(angle_range=(-30, 30), p=p_s)),
        #     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=p_s),  # thay đổi tương phản rõ rệt hơn

        # ])
        p = 0.3
        self.weak_augment = A.Compose([
            A.Resize(img_size, img_size),
            A.ElasticTransform(
                alpha=10, 
                sigma=2,   
                p=p
            ),
        ])
        self.strong_augment = A.Compose([
            A.Resize(img_size, img_size),
            A.ElasticTransform(
                alpha=30, 
                sigma=4,   
                p=p
            ),
        ])
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample_name = self.dataset[index]
        img_path = os.path.join(self.root_dir, f'images/{sample_name}')
        img_data = np.load(img_path)

        # Áp dụng augmentations
        img_w = self.weak_augment(image=img_data.astype('uint8'))['image']
        img_s = self.strong_augment(image=img_w.astype('uint8'))['image']

        img_w = norm01(img_w)
        img_s = norm01(img_s)

        img_w = torch.from_numpy(img_w).float().permute(2, 0, 1)
        img_s = torch.from_numpy(img_s).float().permute(2, 0, 1)
        org_img = img_w.clone()

        return {
            'id': index,
            'img_w': img_w,
            'img_s': img_s,
            'org_img': org_img,
        }

    def __len__(self):
        return self.num_samples


# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, dataset, img_size, use_aug=False, data_path='./proceeded_data/'):
#         super(Dataset, self).__init__()

#         self.dataset = dataset
#         self.root_dir = data_path
#         self.use_aug = use_aug
#         self.num_samples = len(self.dataset)

#         p = 0.3
#         # Định nghĩa các phép augmentations
#         self.aug_transf = A.Compose([
#             A.Resize(img_size, img_size),
#             A.ElasticTransform(
#                 alpha=10,  # Reduced from 20
#                 sigma=2,   # Reduced from 3
#                 p=p
#             ),
#             A.OneOf([
#                 A.GaussNoise(var_limit=(5.0, 20.0), p=p),  # Reduced noise range
#                 A.GaussianBlur(blur_limit=(3, 5), p=p),    # Reduced blur range
#                 A.MedianBlur(blur_limit=3, p=p)            # Reduced blur limit
#             ], p=p),
#             A.OneOf([
#                 A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=p),  # Reduced limits
#                 A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=p)                        # Reduced clip limit
#             ], p=p)
#         ])
        
#         # Định nghĩa các phép transform cơ bản
#         self.transf = A.Compose([
#             A.Resize(img_size, img_size),
#         ])

#     def __getitem__(self, index):
#         if torch.is_tensor(index):
#             index = index.tolist()

#         sample_name = self.dataset[index]
#         img_path = os.path.join(self.root_dir, f'images/{sample_name}')
#         label_path = os.path.join(self.root_dir, f'labels/{sample_name}')

#         # Load npy image and label
#         img_data = np.load(img_path)  # Could be (H, W), (H, W, 3), or (3, H, W)
#         label_data = np.load(label_path)  # Usually (H, W) or (1, H, W)

#         print(f"Image shape: {img_data.shape}")
#         print(f"Label shape: {label_data.shape}")

#         # Kiểm tra và chuẩn hóa ảnh
#         if img_data.ndim == 2:
#             # Chuyển ảnh grayscale (H, W) thành RGB (H, W, 3)
#             img_data = np.stack([img_data] * 3, axis=-1)
#         elif img_data.ndim == 3:
#             # Nếu ảnh có 3 kênh nhưng trong dạng (3, H, W), chuyển thành (H, W, 3)
#             if img_data.shape[0] == 3 and img_data.shape[2] != 3:
#                 img_data = np.transpose(img_data, (1, 2, 0))  # (3, H, W) → (H, W, 3)
#             elif img_data.shape[2] != 3:
#                 raise ValueError(f"Expected image to have 3 channels, but got {img_data.shape[2]} channels.")
#         else:
#             raise ValueError(f"Unexpected image shape: {img_data.shape}")

#         # Kiểm tra hình ảnh sau khi chuyển đổi
#         print(f"Post-processed image shape: {img_data.shape}")

#         # Kiểm tra nhãn
#         if label_data.ndim == 3:
#             label_data = label_data.squeeze()  # (1, H, W) → (H, W)
#         if label_data.ndim != 2:
#             raise ValueError(f"Unsupported label shape: {label_data.shape}")

#         # Áp dụng augmentations
#         if self.use_aug:
#             tsf = self.aug_transf(image=img_data.astype('uint8'), mask=label_data.astype('uint8'))
#         else:
#             tsf = self.transf(image=img_data.astype('uint8'), mask=label_data.astype('uint8'))

#         img_data, label_data = tsf['image'], tsf['mask']

#         # Chuẩn hóa ảnh về [0, 1] và xử lý nhãn
#         img_data = norm01(img_data)
#         label_data = process_multiclass_label(label_data)

#         # Chuyển thành tensor
#         img_data = torch.from_numpy(img_data).float()          # (H, W, 3)
#         label_data = torch.from_numpy(label_data).float()      # (H, W) or (H, W, C)

#         img_data = img_data.permute(2, 0, 1)  # → (3, H, W)

#         org_img = img_data.clone()

#         print(f"Tensor image shape: {img_data.shape}")
#         print(f"Tensor label shape: {label_data.shape}")

#         return {
#             'image': img_data,
#             'label': label_data,
#             'org_img': org_img,
#         }

#     def __len__(self):
#         return self.num_samples


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, img_size, use_aug=False, data_path='./proceeded_data/'):
        super(Dataset, self).__init__()
        
        self.dataset = dataset
        self.root_dir = data_path
        self.use_aug = use_aug

        self.num_samples = len(self.dataset)

        p = 0.3
        self.aug_transf = A.Compose([
            A.Resize(img_size, img_size),
            A.ElasticTransform(
                alpha=10,  # Reduced from 20
                sigma=2,   # Reduced from 3
                p=p
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 20.0), p=p),  # Reduced noise range
                A.GaussianBlur(blur_limit=(3, 5), p=p),    # Reduced blur range
                A.MedianBlur(blur_limit=3, p=p)            # Reduced blur limit
            ], p=p),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=p),  # Reduced limits
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=p)                        # Reduced clip limit
            ], p=p)
        ])
        self.transf = A.Compose([
            A.Resize(img_size, img_size),
        ])
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
    

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample_name = self.dataset[index]
        img_path = os.path.join(self.root_dir, f'images/{sample_name}')
        label_path = os.path.join(self.root_dir, f'labels/{sample_name}')

        img_data = np.load(img_path)
        label_data = np.load(label_path)
        # img_data = np.transpose(img_data, (1, 2, 0))


        if self.use_aug:
            tsf = self.aug_transf(image=img_data.astype('uint8'), mask=label_data.astype('uint8'))
        else:
            tsf = self.transf(image=img_data.astype('uint8'), mask=label_data.astype('uint8'))
        img_data, label_data = tsf['image'], tsf['mask']
        
        # Bỏ việc chia cho 255 ở đây vì norm01 đã làm việc này rồi
        img_data = norm01(img_data)
        label_data = process_multiclass_label(label_data)

        img_data = torch.from_numpy(img_data).float()
        label_data = torch.from_numpy(label_data).float()

        img_data = img_data.permute(2, 0, 1)
        org_img = img_data
        # img_data = self.normalize(img_data)

        return{
            'image': img_data,
            'label': label_data,
            'org_img': org_img,
        }


    def __len__(self):
        return self.num_samples
    
    
class SWDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, img_size, use_aug=False, data_path='./proceeded_data/'):
        super(SWDataset, self).__init__()
        
        self.dataset = dataset
        self.root_dir = data_path
        self.use_aug = use_aug
        self.img_size = img_size
        
        self.num_samples = len(self.dataset)
        
        p_w = 0.3
        p_s = 0.7
        
        # Strong augmentation
        self.strong_aug = A.Compose([
            A.Resize(img_size, img_size),
            A.OneOf([
                A.RandomRotate90(p=p_s),
                A.HorizontalFlip(p=p_s),
                A.VerticalFlip(p=p_s)
            ], p=p_s),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=p_s),
                A.GaussianBlur(blur_limit=(3, 7), p=p_s),
            ], p=p_s),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=p_s),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=p_s)
            ], p=p_s)
        ])
        
        # Weak augmentation
        self.weak_aug = A.Compose([
            A.Resize(img_size, img_size),
            A.ElasticTransform(
                alpha=10,  # Reduced from 20
                sigma=2,   # Reduced from 3
                p=p_w
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 20.0), p=p_w),  # Reduced noise range
                A.GaussianBlur(blur_limit=(3, 5), p=p_w),    # Reduced blur range
                A.MedianBlur(blur_limit=3, p=p_w)            # Reduced blur limit
            ], p=p_w),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=p_w),  # Reduced limits
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=p_w)                        # Reduced clip limit
            ], p=p_w)   
        ])

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        sample_name = self.dataset[index]
        img_path = os.path.join(self.root_dir, f'images/{sample_name}')
        label_path = os.path.join(self.root_dir, f'labels/{sample_name}')

        img_data = np.load(img_path)
        label_data = np.load(label_path)
        
        strong_tsf = self.strong_aug(image=img_data.astype('uint8'), mask=label_data.astype('uint8'))
        strong_img, strong_label = strong_tsf['image'], strong_tsf['mask']
            
        weak_tsf = self.weak_aug(image=img_data.astype('uint8'), mask=label_data.astype('uint8'))
        weak_img, weak_label = weak_tsf['image'], weak_tsf['mask']

        # Normalize images
        strong_img = norm01(strong_img)
        weak_img = norm01(weak_img)
        
        # Process labels
        strong_label = process_multiclass_label(strong_label)
        weak_label = process_multiclass_label(weak_label)

        # Convert to tensors
        strong_img = torch.from_numpy(strong_img).float().permute(2, 0, 1)
        weak_img = torch.from_numpy(weak_img).float().permute(2, 0, 1)
        strong_label = torch.from_numpy(strong_label).float()
        weak_label = torch.from_numpy(weak_label).float()

        return {
            'img_s': strong_img,
            'img_w': weak_img,
            'label_s': strong_label,
            'label_w': weak_label,
            'org_img': weak_img.clone()
        }
        
    def __len__(self):
        return self.num_samples


def get_dataset(args, img_size=384, supervised_ratio=0.2, train_aug=False, k=6, lb_dataset=Dataset, ulb_dataset=StrongWeakAugment, v_dataset=Dataset):
    """
    Creates datasets by splitting data into labeled, unlabeled and validation sets.
    
    Args:
        args: Configuration arguments
        img_size: Size to resize images to
        supervised_ratio: Ratio of labeled to total training data
        train_aug: Whether to use data augmentation for training
        k: Which fold to use for validation (1-5)
        lb_dataset: Dataset class for labeled data
        ulb_dataset: Dataset class for unlabeled data 
        v_dataset: Dataset class for validation data
        
    Returns:
        Dictionary containing labeled, unlabeled and validation datasets
    """
    # Load fold data
    folds = []
    for idx in range(1, 6):
        with open(f'{args.data.train_folder}/fold{idx}.txt', 'r') as f:
            fold = [line.strip() for line in f.readlines()]
        folds.append(fold)

    # Combine training folds (excluding validation fold)
    train_data = []
    for j in range(5):
        if j != k - 1:
            train_data.extend(folds[j])
    train_data = sorted(train_data)

    if not train_data:
        raise ValueError("No training data found.")

    # Split into labeled and unlabeled
    l_data = sorted(random.sample(train_data, int(len(train_data) * supervised_ratio)))
    u_data = sorted(list(set(train_data) - set(l_data)))

    # Create datasets
    l_dataset = lb_dataset(dataset=l_data, img_size=img_size, use_aug=train_aug, data_path=args.data.train_folder)
    u_dataset = ulb_dataset(dataset=u_data, img_size=img_size, use_aug=train_aug, data_path=args.data.train_folder)

    # Get validation data
    val_data = sorted(folds[k - 1])
    if not val_data:
        raise ValueError(f"No validation data found in fold {k}.")

    val_dataset = v_dataset(dataset=val_data, img_size=img_size, use_aug=False, data_path=args.data.val_folder)

    # Print dataset info
    print(f'Train Data: {train_data[0]} - {len(train_data)}')
    print(f'Labeled Data: {l_data[0] if l_data else "None"} - {len(l_data)}')
    print(f'Unlabeled Data: {u_data[0] if u_data else "None"} - {len(u_data)}')
    print(f'Val Data: {val_data[0]} - {len(val_data)}')

    return {
        'lb_dataset': l_dataset,
        'ulb_dataset': u_dataset,
        'val_dataset': val_dataset
    }


def get_dataset_without_full_label(args, img_size=384, train_aug=False, k=6, lb_dataset=Dataset, ulb_dataset=StrongWeakAugment, v_dataset=Dataset):
    """
    Creates datasets from pre-split labeled and unlabeled data.
    
    Args:
        args: Arguments containing data paths
        img_size: Size to resize images to
        train_aug: Whether to use data augmentation for training
        k: Fold number to use for validation (1-5)
        lb_dataset: Dataset class for labeled data
        ulb_dataset: Dataset class for unlabeled data
        v_dataset: Dataset class for validation data
        
    Returns:
        Dictionary containing labeled, unlabeled and validation datasets
    """
    # Load fold data
    folds = []
    for idx in range(1, 6):
        with open(f'{args.data.train_folder}/fold_label_{idx}.txt', 'r') as f:
            fold_data = [line.strip() for line in f.readlines()]
        folds.append(fold_data)
            
    # Get training data from all folds except validation fold
    l_data = []
    for j in range(5):
        if j != k - 1:
            l_data.extend(folds[j])
    l_data = sorted(l_data)
    
    # Load unlabeled data
    with open(f'{args.data.unlabeled_folder}/unlabeled.txt', 'r') as f:
        u_data = sorted([line.strip() for line in f.readlines()])

    # Validate data
    if not l_data:
        raise ValueError("No labeled training data found.")
    if not u_data:
        raise ValueError("No unlabeled training data found.")
    
    # Create datasets
    l_dataset = lb_dataset(dataset=l_data, img_size=img_size, use_aug=train_aug, data_path=args.data.train_folder)
    u_dataset = ulb_dataset(dataset=u_data, img_size=img_size, use_aug=train_aug, data_path=args.data.unlabeled_folder)

    # Get validation data from fold k
    val_data = sorted(folds[k - 1])
    if not val_data:
        raise ValueError(f"No validation data found in fold {k}.")

    val_dataset = v_dataset(dataset=val_data, img_size=img_size, use_aug=False, data_path=args.data.val_folder)

    # Print dataset info
    print(f'Labeled Data: {l_data[0]} - {len(l_data)}')
    print(f'Unlabeled Data: {u_data[0]} - {len(u_data)}')
    print(f'Val Data: {val_data[0]} - {len(val_data)}')
    
    return {
        'lb_dataset': l_dataset,
        'ulb_dataset': u_dataset,
        'val_dataset': val_dataset
    }

def get_dataset_without_full_label_without_val(args, img_size=384, train_aug=False, k=6, lb_dataset=Dataset, ulb_dataset=StrongWeakAugment, v_dataset=Dataset):
    """
    Creates datasets from pre-split labeled and unlabeled data, using training data for validation.
    
    Args:
        args: Arguments containing data paths
        img_size: Size to resize images to
        train_aug: Whether to use data augmentation for training
        k: Unused fold parameter (kept for API consistency)
        lb_dataset: Dataset class for labeled data
        ulb_dataset: Dataset class for unlabeled data
        v_dataset: Unused dataset class (kept for API consistency)
        
    Returns:
        Dictionary containing labeled and unlabeled datasets
    """
    # Load fold data
    folds = []
    for idx in range(1, 6):
        with open(f'{args.data.train_folder}/fold_label_{idx}.txt', 'r') as f:
            fold_data = [line.strip() for line in f.readlines()]
        folds.append(fold_data)
            
    # Get all labeled data from folds
    l_data = []
    for fold in folds:
        l_data.extend(fold)
    l_data = sorted(l_data)
    
    # Load unlabeled data
    with open(f'{args.data.unlabeled_folder}/unlabeled.txt', 'r') as f:
        u_data = sorted([line.strip() for line in f.readlines()])

    # Validate data
    if not l_data:
        raise ValueError("No labeled training data found.")
    if not u_data:
        raise ValueError("No unlabeled training data found.")
    
    # Create datasets
    l_dataset = lb_dataset(dataset=l_data, img_size=img_size, use_aug=train_aug, data_path=args.data.train_folder)
    u_dataset = ulb_dataset(dataset=u_data, img_size=img_size, use_aug=train_aug, data_path=args.data.unlabeled_folder)

    # Print dataset info
    print(f'Labeled Data: {l_data[0]} - {len(l_data)}')
    print(f'Unlabeled Data: {u_data[0]} - {len(u_data)}')
    
    return {
        'lb_dataset': l_dataset,
        'ulb_dataset': u_dataset,
    }

class ABDDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, img_size, use_aug=False, data_path='./proceeded_data/'):
        super(ABDDataset, self).__init__()
        
        self.dataset = dataset
        self.root_dir = data_path
        self.use_aug = use_aug
        self.img_size = img_size
        
        self.num_samples = len(self.dataset)
        
        p_w = 0.3
        p_s = 0.7
        
        # Strong augmentation
        self.strong_aug = A.Compose([
            A.Resize(img_size, img_size),
            A.OneOf([
                A.RandomRotate90(p=p_s),
                A.HorizontalFlip(p=p_s),
                A.VerticalFlip(p=p_s)
            ], p=p_s),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=p_s),
                A.GaussianBlur(blur_limit=(3, 7), p=p_s),
            ], p=p_s),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=p_s),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=p_s)
            ], p=p_s)
        ])
        
        # Weak augmentation
        self.weak_aug = A.Compose([
            A.Resize(img_size, img_size),
            A.ElasticTransform(
                alpha=10,
                sigma=2,   
                p=p_w
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 20.0), p=p_w),
                A.GaussianBlur(blur_limit=(3, 5), p=p_w),
                A.MedianBlur(blur_limit=3, p=p_w)
            ], p=p_w),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=p_w),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=p_w)
            ], p=p_w)   
        ])

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        sample_name = self.dataset[index]
        img_path = os.path.join(self.root_dir, f'images/{sample_name}')
        label_path = os.path.join(self.root_dir, f'labels/{sample_name}')

        img_data = np.load(img_path)
        label_data = np.load(label_path)
        
        # Apply strong augmentation
        strong_tsf = self.strong_aug(image=img_data.astype('uint8'), mask=label_data.astype('uint8'))
        strong_img, strong_label = strong_tsf['image'], strong_tsf['mask']
            
        # Apply weak augmentation
        weak_tsf = self.weak_aug(image=img_data.astype('uint8'), mask=label_data.astype('uint8'))
        weak_img, weak_label = weak_tsf['image'], weak_tsf['mask']

        # Normalize images
        strong_img = norm01(strong_img)
        weak_img = norm01(weak_img)
        
        # Process labels
        strong_label = process_multiclass_label(strong_label)
        weak_label = process_multiclass_label(weak_label)

        # Convert to tensors
        strong_img = torch.from_numpy(strong_img).float().permute(2, 0, 1)
        weak_img = torch.from_numpy(weak_img).float().permute(2, 0, 1)
        strong_label = torch.from_numpy(strong_label).float()
        weak_label = torch.from_numpy(weak_label).float()

        return {
            'image': weak_img,
            'image_strong': strong_img,
            'label': weak_label,
            'label_strong': strong_label,
            'id': index
        }

    def __len__(self):
        return self.num_samples
