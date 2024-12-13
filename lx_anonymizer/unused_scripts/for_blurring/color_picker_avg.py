import cv2
import numpy as np
from sklearn.cluster import KMeans

def get_dominant_color(image, box, k=1):
    # Crop the background area around the text box
    (startX, startY, endX, endY) = box
    margin = 10  # You can adjust the size of the margin
    cropped_img = image[max(startY - margin, 0):min(endY + margin, image.shape[0]),
                        max(startX - margin, 0):min(endX + margin, image.shape[1])]
    
    # Convert to a color space that may improve color clustering
    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2LAB)

    # Reshape the image to be a list of pixels
    pixels = cropped_img.reshape((cropped_img.shape[0] * cropped_img.shape[1], 3))

    # Use KMeans clustering to find the most prevalent color
    clt = KMeans(n_clusters=k, n_init=10)
    clt.fit(pixels)

    # Find the most prevalent cluster center
    numLabels = np.arange(0, k + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()

    # Find the largest cluster
    dominant_color = clt.cluster_centers_[np.argmax(hist)]

    # Convert dominant color back to BGR color space for OpenCV to display
    dominant_color = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_LAB2BGR)[0][0]
    
    return tuple(map(int, dominant_color))
