import cv2
import numpy as np


from sklearn.cluster import DBSCAN

def non_maximum_suppression(coords, dist_thresh):
    pom = []
    coords = np.array(coords)
    clustering = DBSCAN(eps=dist_thresh, min_samples=1).fit(coords)
    labels = clustering.labels_
    unique_labels = set(labels)
    centers = np.array([coords[labels == label].mean(axis=0) for label in unique_labels])
    
    for i in centers:
        pom.append((int(i[0]), int(i[1])))

    return pom


def harris_corner_detection(image, point_size=5, margin=175):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    blockSize = 2
    ksize = 3
    k = 0.04
    zoz = []

    corners = cv2.cornerHarris(gray, blockSize, ksize, k)
    corners = cv2.dilate(corners, None)
    threshold = 0.001 * corners.max()

    height, width = image.shape[:2]

    limited_corners = corners[margin:height-margin, margin:width-margin]
    corner_coordinates = np.column_stack(np.where(limited_corners > threshold))
    
    for coord in corner_coordinates:
        x, y = coord + margin
        zoz.append((y,x))
        cv2.circle(image, (y, x), point_size, (0, 255, 0), -1)
    
    return zoz, image

i1 = 'nazov1.tif'
i2 = 'nazov2.tif'

image1 = cv2.imread(i1)
image2 = cv2.imread(i2)

inverse_difference = cv2.absdiff(image1, image2)
inverse_difference = 255 - inverse_difference

zoz, image = harris_corner_detection(inverse_difference.copy(), point_size=10)
dist_thresh = 10  
filtered_corners = non_maximum_suppression(zoz, dist_thresh)
point_size = 2

for i in filtered_corners:
    cv2.circle(image, (int(i[0]), int(i[1])), point_size, (255, 0, 0), -1)


from scipy.optimize import linear_sum_assignment

def connect_regions(image1, regions1):
    distances = np.zeros((len(regions1), len(regions1)))
    for i, region1 in enumerate(regions1):
        for j, region2 in enumerate(regions1):
            if i != j:
                distances[i, j] = np.linalg.norm(np.array(region1) - np.array(region2))
                distances[j, i] = np.linalg.norm(np.array(region1) - np.array(region2))
            else:
                distances[i, j] = np.inf

    row_ind, col_ind = linear_sum_assignment(distances)

    for i, j in zip(row_ind, col_ind):
        cv2.line(image, regions1[i], regions1[j], (0, 255, 0), 2) 

    return image1

prem = connect_regions(image, filtered_corners)

cv2.imwrite(f'vysledok.jpg', prem)



