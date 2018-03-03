import h5py
import numpy as np
from skimage.morphology import watershed
from skimage.filters import threshold_otsu, sobel
from skimage.morphology import opening, disk, reconstruction
from skimage.measure import label
from copy import deepcopy
import sys
import math


# Utility function to load images from a HDF5 file
def load_images(filename):
    image_key = "images"
    label_key = "CellType"
    images = []
    with h5py.File(filename) as f:
        image_set = f[image_key]
        labels = [label for label in f[label_key]]
        for image in image_set:
            images.append(image)
    return np.array(images), labels


# Method to label possible nuclei from images
def get_elevation_based_labels(image):
    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    mask = image
    dilated = reconstruction(seed, mask, method='dilation')
    hdome = image - dilated

    elevation_map = sobel(hdome)
    elevation_thresh = threshold_otsu(elevation_map)
    markers = np.where(elevation_map >= elevation_thresh, 2, 0)

    pixel_values = list(set(hdome.flatten()))
    lower_thresh = int(np.percentile(pixel_values, 1))
    markers[hdome <= lower_thresh] = 1

    segmentation = watershed(elevation_map, markers)

    selem = disk(6)
    labels = opening(segmentation, selem)
    return label(labels)


# Our complete method to segment nuclei from images
def watershed_segmentation(image, img_number):
    im_width, im_height = image.shape

    labels = get_elevation_based_labels(image)

    # Map regions to coordinates
    box_coords = {}
    d, k = labels.shape
    for i in range(d):
        for j in range(k):
            if labels[i][j] in box_coords:
                box_coords[labels[i][j]].append((i,j))
            else:
                box_coords[labels[i][j]] = [(i,j)]

    # Generating new cropped images
    new_images = []
    max_len_region = max([len(val) for val in box_coords.values()])

    for key in box_coords.keys():
        region = box_coords[key]

        # Ignore the background tag
        if len(region) == max_len_region:
            continue

        # Found Basin
        xs, ys = zip(*region)
        minx = min(xs)
        maxx = max(xs)
        miny = min(ys)
        maxy = max(ys)

        optimums = fix_corner_nuclei(minx, maxx, miny, maxy,
                                     im_width, im_height)
        (minx, maxx, miny, maxy) = optimums

        # Ignore segmentations that are too small
        if (maxx - minx) < 50 and (maxy - miny) < 50:
            continue

        new_image = deepcopy(image[minx:maxx+1, miny:maxy+1])

        area = 0.0

        # Zero out non-key region pixels
        region_keys = set(region)
        for i in range(minx, maxx + 1):
            for j in range(miny, maxy + 1):
                if (i, j) not in region_keys:
                    new_image[i - minx, j - miny] = 0.0
                else:
                    area += 1.
        total = (maxx - minx) * (maxy - miny)

        # If you want to print the area of the nucleus vs. area of background
        #print("Region area: ", area, total)

        new_images.append((new_image, area))

        # Print number of nuclei processed and pixel locations for each
        print("Nuclei Number:", img_number + len(new_images),
              "Rectangle", minx, maxx, miny, maxy)

    return new_images


# Trim the edge blur off an edge nucleus
def fix_corner_nuclei(minx, maxx, miny, maxy, width, height):
    # width should be 0 - 511, height should be 0 - 511
    padding = 15  # Nuclei should be 15 pixels within image
    bad_top_edge = 0 <= minx < padding
    bad_bottom_edge = width - padding <= maxx < width
    bad_left_edge = 0 <= miny < padding
    bad_right_edge = height - padding <= maxy < height

    if bad_top_edge:
        minx = padding
    if bad_bottom_edge:
        maxx = width - padding
    if bad_left_edge:
        miny = padding
    if bad_right_edge:
        maxy = height - padding

    return (minx, maxx, miny, maxy)


# Go through the images and generate nuclei crops
def generate_crops(image_set, labels):
    cropped_imgs = []
    count = 0

    for idx in range(len(image_set)):
        image = image_set[idx]
        label = labels[idx]
        count += 1
        if np.max(image) >= 1000:  # Get rid of images with no nuclei
            new_crops = watershed_segmentation(image, len(cropped_imgs))
            for img, area in new_crops:
                cropped_imgs.append((img, area, label))

        print("PROGRESS", count, len(image_set))

    cropped_imgs, crop_labels = pad_crops(cropped_imgs)

    # After generating full nuclei images, also extract
    # patches from the nuclei
    patches = []
    patch_labels = []

    for idx in range(len(cropped_imgs)):
        crop = cropped_imgs[idx]
        crop_label = crop_labels[idx]
        new_patches, new_labels = get_patches(crop, label)
        if len(new_patches) > 0:
            patches += new_patches
            patch_labels += new_labels

    return np.array(cropped_imgs), crop_labels, np.array(patches), patch_labels


# Get 32 x 32 patches from a segmented nucleus
def get_patches(crop, label):
    patch_size = 32
    r, c = crop.shape
    new_patches = []
    new_labels = []
    j = 0
    while j < r - patch_size:
        i = 0
        while i < c - patch_size:
            new_patch = deepcopy(crop[i:i+patch_size, j:j+patch_size])
            area = np.sum(new_patch > 0)
            # Keep only patches containing mostly nucleus regions
            if area >= .8 * patch_size * patch_size:
                if overexposed_image(new_patch, area):
                    print("OVEREXPOSED PATCH")
                else:
                    new_patches.append(new_patch)
                    new_labels.append(label)
            i += int(patch_size / 2)
        j += int(patch_size / 2)

    return new_patches, new_labels


# Pad the segmentations so that they are all the same size (128 x 128)
def pad_crops(cropped_images):
    new_crops = []
    avg_area = sum([area for (image, area, label) in cropped_images]) / len(cropped_images)
    std = sum([(area - avg_area) ** 2 for (image, area, label) in cropped_images]) / len(cropped_images)
    std = math.sqrt(std)
    #print(avg_area, std)
    max_r = 0.0
    max_c = 0.0
    crop_r = 128
    crop_c = 128
    areas = [area for (image, area, label) in cropped_images]

    good_crops = []
    good_labels = []
    tossed_images = []
    large_images = []
    for image, area, label in cropped_images:
        z_score = (area- avg_area) / std
        if area <= 2000:  # cut out small bright spots
            tossed_images.append((image, "AREA TOO SMALL"))
            continue
        else:
            r, c = image.shape
            if r > crop_r or c > crop_c:
                tossed_images.append((image, "MARGINS TOO LARGE"))
                large_images.append(image)
                continue
            if overexposed_image(image, area):
                tossed_images.append((image, "OVEREXPOSED"))
                print("OVEREXPOSED")
                continue
            max_r = max(max_r, r)
            max_c = max(max_c, c)
            good_crops.append(image)
            good_labels.append(label)
    #print(max_r, max_c)
    #f = open("tossed_images.p", 'wb')
    #pickle.dump(tossed_images, f)
    for image in good_crops:
        new_crops.append(pad_nuclei(image, crop_r, crop_c))
        #new_crops.append(pad_nuclei(image, max_r, max_c))

    return new_crops, good_labels


# Check if the pixel intesities are too high in an image
def overexposed_image(image, area):
    # working with 12 bit images
    threshold = 3700
    exposure_filter = np.where(image > threshold, 1., 0.)
    if np.sum(exposure_filter) >= .25 * area:
        return True
    return False


def pad_nuclei(image, max_r, max_c):
    r, c = image.shape
    shift_r = int((max_r - r) / 2)
    shift_c = int((max_c - c) / 2)

    new_image = np.zeros((max_r, max_c))
    new_image[shift_r: shift_r + r, shift_c: shift_c + c] = image

    return new_image


def main(args):

    fname = args[1]
    image_set, labels = load_images(fname)
    cropped_images, crop_labels, patch_images, patch_labels = generate_crops(image_set, labels)

    oname = fname[:-5] + "_crops.hdf5"
    opname = fname[:-5] + "_patch_crops.hdf5"
    with h5py.File(oname, "w") as f:
        f.create_dataset('images', data=cropped_images)
        f.create_dataset('CellType', data=crop_labels)
    with h5py.File(opname, "w") as f:
        f.create_dataset('images', data=patch_images)
        f.create_dataset('CellType', data=patch_labels)

if __name__ == "__main__":
    main(sys.argv)
