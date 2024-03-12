import random
import numpy as np

def flip_augmentation(images, labels, aug_ratio = 0.2):
    """
    Method to augment the dataset size by flipping some images (randomly chosen) 
    in the dataset. We take a aug_ratio*100 percent of images for each label and 
    flip them.
    inputs:
        images: images in the dataset
        labels: labels of the images
        aug_ratio: ratio of images to flip
    output:
        augmented_images: images after augmentation
        augmented_labels: labels after augmentation
    """

    labels_ = np.unique(labels)
    augmented_images = []
    augmented_labels = []

    for label in labels_:
        idx_label = np.where(labels == label)[0]
        num_aug = int(len(idx_label)*aug_ratio)
        idx_aug = random.sample(list(idx_label), num_aug)

        for idx in idx_aug:
            image = images[idx]

            image = np.reshape(image, (3, 32,32))

            red_channel = image[0]
            green_channel = image[1]
            blue_channel = image[2]

            new_image = np.dstack((red_channel, green_channel, blue_channel))

            new_image = np.fliplr(new_image)

            red_chanel = new_image[:,:,0]
            green_chanel = new_image[:,:,1]
            blue_chanel = new_image[:,:,2]

            new_image = np.array([red_chanel,green_chanel,blue_chanel])

            aufmented_images.append(new_image)
            augmented_labels.append(label)
    
    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)
    perm = np.random.permutation(len(aufmented_images))

    augmented_images = augmented_images[perm]
    augmented_labels = augmented_labels[perm]

    augmented_images = np.append(images, augmented_images)
    augmented_labels = np.append(labels, augmented_labels)

    perm = np.random.permutation(len(aufmented_images))

    augmented_images = augmented_images[perm]
    augmented_labels = augmented_labels[perm]
    
    return aufmented_images, augmented_labels

