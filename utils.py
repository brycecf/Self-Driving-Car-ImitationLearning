import numpy as np
from PIL import Image, ImageEnhance


def data_generator(img_paths, labels, batch_size=64, is_training=True):
    """
    Data generator for training and validation data.
    
    Args:
        img_paths: Paths for the image files.
        labels: Labels for the training data.
        batch_size: Specifies the resulting array size.
        is_training: Flag to determine if generator is being used for training or validation.
    
    Returns:
        Yields originals training images and copies with different brightness levels.
    """
    def process_image(img_path, label, correction, num_transforms=5):
        orig_img = Image.open(img_path)
        transformed_imgs = []
        label = [label + correction]
        if is_training:
            for brightness in np.random.normal(0.5, 0.35, size=num_transforms):
                transformed_imgs.append(np.array(ImageEnhance.Brightness(orig_img).enhance(brightness)))
            label = label * (num_transforms + 1)
        transformed_imgs.append(np.array(orig_img))
        return transformed_imgs, label 
    
    while True:
        batch_features = []
        batch_labels = []
        for _ in range(batch_size):
            i = np.random.choice(len(img_paths), 1)[0]
            center_features, center_labels = process_image(img_paths['Center'].iloc[i], labels.iloc[i], 0)
            left_features, left_labels = process_image(img_paths['Left'].iloc[i], labels.iloc[i], 0.175)
            right_features, right_labels = process_image(img_paths['Right'].iloc[i], labels.iloc[i], -0.15)
            batch_features += center_features + left_features + right_features
            batch_labels += center_labels + left_labels + right_labels
        yield np.array(batch_features), np.array(batch_labels, dtype=np.float64)