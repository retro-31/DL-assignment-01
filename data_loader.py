from keras.datasets import fashion_mnist, mnist # type: ignore
import numpy as np
import matplotlib.pyplot as plt

def load_data(dataset="fashion_mnist"):
    if dataset == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    else:
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    # unique classes
    classes = np.unique(y_train)

    # sample image from each class
    sample_images = []
    for cls in classes:
        idx = np.where(y_train == cls)[0][0]
        sample_images.append(X_train[idx])

    # plotting images
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    axes = axes.flatten()

    for i, img in enumerate(sample_images):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Class {classes[i]}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    X_train = X_train.reshape(-1, 784) / 255.0
    X_test = X_test.reshape(-1, 784) / 255.0
    return (X_train, y_train), (X_test, y_test)
