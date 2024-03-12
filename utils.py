import random
import numpy as np
from scipy.ndimage import rotate

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

    augmented_images = []
    augmented_labels = []

    for label in set(labels):
        idx_label = np.where(labels == label)[0]
        idx_to_augment = random.sample(list(idx_label), int(len(idx_label)*aug_ratio), replace=False)

        for idx in idx_to_augment:
            image = images[idx]

            image = np.reshape(image, (3, 32,32))
            new_image = np.dstack((image[0], image[1], image[2]))

            new_image = np.fliplr(new_image)

            new_image = np.array([new_image[:,:,0], new_image[:,:,1], new_image[:,:,2]])
            new_image = new_image.flatten()
            augmented_images.append(new_image)
            augmented_labels.append(label)
    
    #To not append in the same order
    ids = np.random.permutation(len(augmented_labels))
    augmented_images = np.array(augmented_images)[ids]
    augmented_labels = np.array(augmented_labels)[ids]

    #To not have images and its augmented versions concatenate
    final_images = np.concatenate((images, augmented_images), axis = 0)
    final_labels = np.concatenate((labels, augmented_images), axis=0)
    ids = np.random.permutation(len(final_labels))
    final_images = final_images[ids]
    final_labels = final_labels[ids]
    
    return final_images, final_labels


def rotate_dataset(images, labels, n_rotations = 1, ratio=0.2, rotate_angle=1):
    '''
    Method to augment the dataset size by rotating some images (randomly chosen)
    in the dataset. We take a ratio*100 percent of images for each label and
    rotate them.

    inputs:
        images: images in the dataset
        labels: labels of the images
        n_rotations: number of rotations per image randomly chosen
        ratio: ratio of images to rotate per label
        rotate_angle: angle to rotate

    output:
        final_images: images after augmentation
        final_labels: labels after augmentation
    '''

    augmented_images = []
    augmented_labels = []
    for i in range(n_rotations):
        for label in set(labels):
            #Get the index where label is
            idx_label = np.where(labels == label)[0]
            #Choose randomly 20% of the data with y=label 
            idx_to_augment = random.sample(list(idx_label), int(len(idx_label)*ratio), replace=False)

            X_to_augment = images[idx_to_augment]
            
            for id in idx_to_augment:
                image = images[id]
                image = np.reshape(image, (3, 32,32))
                x_aux = np.dstack((image[0], image[1], image[2]))

                angle = np.random.randint(-rotate_angle, rotate_angle)
                new_image = rotate(x_aux, angle, reshape=False, mode = 'nearest')

                new_image = np.array([new_image[:,:,0], new_image[:,:,1], new_image[:,:,2]])
                new_image = new_image.flatten()
                augmented_images.append(new_image)
                augmented_labels.append(label)

    #To not append in the same order
    ids = np.random.permutation(len(augmented_labels))
    augmented_images = np.array(augmented_images)[ids]
    augmented_labels = np.array(augmented_labels)[ids]

    #To not have images and its augmented versions concatenate
    final_images = np.concatenate((images, augmented_images), axis = 0)
    final_labels = np.concatenate((labels, augmented_images), axis=0)
    ids = np.random.permutation(len(final_labels))
    final_images = final_images[ids]
    final_labels = final_labels[ids]


    return final_images, final_labels
