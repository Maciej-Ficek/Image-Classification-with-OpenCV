import cv2
import numpy as np
import sys
from mpi4py import MPI
from pathlib import Path

def load_grayed_images(path):
    """
    Loads the images
    and turns them to grayscale.
    param path: path to the folder with images.
    """
    folder_path = Path(path);
    filepath = folder_path / 'img001.jpeg';
    get_shapes = np.array(cv2.imread(str(filepath)));

    #I'm not explaining MPI stuff as it is a bonus
    comm = MPI.COMM_WORLD;
    rank = comm.Get_rank();
    size = comm.Get_size();
    chunks = [[] for _ in range(size)];
    for i in range(1, 601, 1):
        chunks[i % size].append(i);
    prefix_sums = [0] + [len(chunk) for chunk in chunks];
    prefix_sums = comm.scan(prefix_sums);
    prefix_sum = prefix_sums[rank];

    result = np.zeros((len(chunks[rank]), get_shapes.shape[0], get_shapes.shape[1]));
    for i, value in enumerate(chunks[rank]):
        filepath = folder_path / f"img{value:03d}.jpeg";
        result[i, : , : ] = cv2.cvtColor(np.array(cv2.imread(str(filepath))), cv2.COLOR_BGR2GRAY);

    results = comm.gather(result, root=0);
    if rank == 0:
        # Combine the results from all processes
        images = np.concatenate(results, axis=0);
        image_indices = [value for chunk in chunks for value in chunk];
        images = images[np.argsort(image_indices)];
        return images;

def get_background(images):
    """
    Takes numpy array containing images
    and returns their background.
    param images: numpy array of images to get background from
    """
    background = np.median(images, axis = 0);
    return background;

def count_measure_from_background(images, background):
    """
    Counts the distance of each image from images
    and background in the sense of absolute value
    of difference between pixel's values
    param images: numpy array containing gray images
    param background: background of elements of images array
    """
    distances = np.zeros((len(images), images.shape[1], images.shape[2]));
    for i in range(0, len(images), 1):
        distances[i, :, :] = np.array(abs(images[i] - background));
    return distances;

def divide_images_into_classes(distances, first_border, second_border, cut_off, activation):
    """
    Takes array of distances between
    images and their background
    and uses it to classify images
    to three classes.
    param distances: numpy array of pixel distances
    param first_border: border between left and center chamber
    param second_border: border between center and right chamber
    param cut_off: the minimum distance of pixel to count it.
    Cut_off protects us from counting small distances of pixels
    (noise between background and the given image)
    as pixels originated from mouse
    param activation: minimum number of pixels with big distance
    to activate classification of the image
    The higher activation, the smaller number of images
    classified to two classes in border cases
    """
    # array containing classification of each image
    # classification[i,j] = 1 iff i-th image is classified to j-th class
    classification = np.zeros((len(distances), 3));
    # list containing sums of images classified to each of three classes
    category = np.zeros((3), dtype='int');
    for i in range(0, len(distances), 1):
        distance_left_chamber = distances[i, :, : first_border];
        mouse_pixels_in_left_chamber = sum(distance_left_chamber[distance_left_chamber >= cut_off]);
        distance_center_chamber = distances[i, :, first_border + 1 : second_border];
        mouse_pixels_in_center_chamber = sum(distance_center_chamber[distance_center_chamber >= cut_off]);
        distance_right_chamber = distances[i, :, second_border + 1 :];
        mouse_pixels_in_right_chamber = sum(distance_right_chamber[distance_right_chamber >= cut_off]);
        if mouse_pixels_in_left_chamber >= activation:
            category[0] += 1;
            classification[i, 0] += 1;
        if mouse_pixels_in_center_chamber >= activation:
            category[1] += 1;
            classification[i, 1] += 1;
        if mouse_pixels_in_right_chamber >= activation:
            category[2] += 1;
            classification[i, 2] += 1;
    return (classification, category);

def display_and_save_results(classification, category):
    """
    Displays results to console and
    saves them in results.txt file.
    param classes: table containing information
    about classification of each image
    param category: list contaning amount
    of images classified to each class
    """
    with open("results.txt", "a") as f:
        for i in range(1, len(classification) + 1, 1):
            #False - the mouse is not in chamber, True - the mouse is in chamber
            print(f"image{i:03d} : {classification[i - 1, 0] == True}, {classification[i - 1, 1] == True}, {classification[i - 1, 2] == True}");
            print(f"image{i:03d} : {classification[i - 1, 0] == True}, {classification[i - 1, 1] == True}, {classification[i - 1, 2] == True}", file=f);
        print(f"Number of images with mouse in first chamber: {category[0]}");
        print(f"Number of images with mouse in second chamber: {category[1]}");
        print(f"Number of images with mouse in third chamber: {category[2]}");
        print(f"Number of images with mouse in first chamber: {category[0]}", file=f);
        print(f"Number of images with mouse in second chamber: {category[1]}", file=f);
        print(f"Number of images with mouse in third chamber: {category[2]}", file=f);

def main(args):
    """
    Required arguments:
    param args[0]: name of the file: ex1.py
    param args[1]: value of border between left and center chamber
    param args[2]: value of border between center and right chamber
    param args[3]: value of cut_off
    param args[4]: value of activation
    param args[5]: path to the folder with images
    """
    images = load_grayed_images(args[5]);
    background = get_background(images);
    distances = count_measure_from_background(images, background);
    results = divide_images_into_classes(distances, int(args[1]), int(args[2]), int(args[3]), int(args[4]));
    display_and_save_results(results[0], results[1]);
    cv2.imshow(images[0, :, :], classification[0, :]);

main(sys.argv);
