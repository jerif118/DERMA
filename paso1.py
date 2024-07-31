import pandas as pd
import os
from PIL import Image
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor

def process_image(idx, row, input_dir, output_dir, num_augmented_images, transform, new_annotations):
    img_name = os.path.join(input_dir, row['isic_id'] + ".jpg")
    if not os.path.exists(img_name):
        return
    
    image = Image.open(img_name).convert('RGB')
    
    for i in range(num_augmented_images):
        augmented_image = transform(image)
        augmented_image = transforms.ToPILImage()(augmented_image)
        augmented_img_name = f'augmented_{row["isic_id"]}_{i}.jpg'
        augmented_image.save(os.path.join(output_dir, augmented_img_name))
        
        new_annotation = row.copy()
        new_annotation['isic_id'] = augmented_img_name.split('.')[0]
        new_annotations.append(new_annotation)
    
    if idx % 100 == 0:
        print(f"imagenes {idx} procesadas")

def save_augmented_images(input_dir, output_dir, annotations_file, num_augmented_images, max_workers=4):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    annotations = pd.read_csv(annotations_file)
    new_annotations = []

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
    ])

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, row in annotations.iterrows():
            executor.submit(process_image, idx, row, input_dir, output_dir, num_augmented_images, transform, new_annotations)

    new_annotations_df = pd.DataFrame(new_annotations)
    new_annotations_df.to_csv('./augmented_train_metadata.csv', index=False)
    print("aumentacion de datos ")

input_dir = './ISIC-imagest'
output_dir = './Augmented-ISIC-images'
annotations_file = './metadata.csv'
num_augmented_images = 5  
max_workers = 8  

save_augmented_images(input_dir, output_dir, annotations_file, num_augmented_images, max_workers)
