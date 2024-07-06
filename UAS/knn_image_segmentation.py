import numpy as np
import cv2
import matplotlib.pyplot as plt

def initialize_centroids(image, k):
    pixels = image.reshape(-1, 3)
    indices = np.random.choice(pixels.shape[0], k, replace=False)
    return pixels[indices]

def assign_clusters(image, centroids):
    pixels = image.reshape(-1, 3)
    distances = np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2)
    clusters = np.argmin(distances, axis=1)
    return clusters

def update_centroids(image, clusters, k):
    pixels = image.reshape(-1, 3)
    new_centroids = np.zeros((k, 3), dtype=np.float32)
    for i in range(k):
        cluster_pixels = pixels[clusters == i]
        if len(cluster_pixels) > 0:
            new_centroids[i] = np.mean(cluster_pixels, axis=0)
    return new_centroids

def kmeans(image, k, max_iters=100):
    centroids = initialize_centroids(image, k)
    for _ in range(max_iters):
        clusters = assign_clusters(image, centroids)
        new_centroids = update_centroids(image, clusters, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids

def segment_image(image, k, max_iters=100):
    clusters, centroids = kmeans(image, k, max_iters)
    segmented_image = centroids[clusters].reshape(image.shape)
    return segmented_image.astype(np.uint8)

def main():
    # Load image
    image_path = 'img/img1.jpg'  # Ganti dengan path gambar Anda
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Parameters
    k = 7  # Jumlah cluster
    max_iters = 100  # Jumlah iterasi maksimum

    # Segment the image
    segmented_image = segment_image(image, k, max_iters)

    # Display the original and segmented image
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f'Segmented Image with {k} Clusters')
    plt.imshow(segmented_image)
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()
