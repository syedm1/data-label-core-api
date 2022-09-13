#!/usr/bin/env python
# coding: utf-8

# @uthors: Sharukh
# Last Update : 29/01/2019
'''
**************************************************************************************************************
Purpose of the file: Generate synthetic card and cash data with annotations
Requirements: Preloaded data packs of card images, cash images and Background images
Features:
    1. Can select type of data( Cash, Card or both; selected will now be refered as data image objects)
    2. Can control transformation of data image objects
    3. Can control resolution of image (default is random sizing of final image)
    4. Generate large data sets based on required parameters
    5. Solves the overlapping card detection for most of the cases
Testing strategy: As the images generated should be checked with annotations some random images are to be
selected from generated data set to ensure consistency
Performance: Varies on number of data image objects included in scene
    1. Two data image objects avg is 6.48s
    2. Six data image objects avg is 18.73s
Areas of improvement:
    1. Using concepts of parallel computing to enhance speed
    2. Generating automated test cases to verify final image and annotation

***************************************************************************************************************
'''

import numpy as np
from glob import glob
from skimage import color, measure
from matplotlib import pyplot
from shapely import affinity
from lxml import etree
import shapely
from shapely.geometry import Polygon, Point, LineString, LinearRing
from imgaug import augmenters as iaa
import imgaug as ia
import pickle
import matplotlib.patches as patches
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import os
import cv2
import time
import argparse
import sys

'''
    Dimensions of the card
'''
cardW = 57
cardH = 87
cornerXmin = 2
cornerXmax = 10.5
cornerYmin = 2.5
cornerYmax = 23

# We convert the measures from mm to pixels: multiply by an arbitrary factor 'zoom'
zoom = 4
cardW *= zoom
cardH *= zoom
cornerXmin = int(cornerXmin * zoom)
cornerXmax = int(cornerXmax * zoom)
cornerYmin = int(cornerYmin * zoom)
cornerYmax = int(cornerYmax * zoom)

data_dir = "data"  # Directory that will contain all kinds of data (the data we download and the data we generate)

if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

card_suits = ['s', 'h', 'd', 'c']
card_values = ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2']

# Pickle file containing the background images
backgrounds_pck_fn = data_dir + "/backgrounds.pck"

# Pickle file containing the card images
cards_pck_fn = data_dir + "/cards.pck"

# Pickle file containing the random card images
random_pck_fn = data_dir + "/random_obj.pck"

# Pickle file containing the cash images
cash_pck_fn = data_dir + "/cash.pck"

refCard = np.array([[0, 0], [cardW, 0], [cardW, cardH], [0, cardH]], dtype=np.float32)
refCardRot = np.array([[cardW, 0], [cardW, cardH], [0, cardH], [0, 0]], dtype=np.float32)
refCornerHL = np.array([[cornerXmin, cornerYmin], [cornerXmax, cornerYmin],
                        [cornerXmax, cornerYmax], [cornerXmin, cornerYmax]], dtype=np.float32)
refCornerLR = np.array([[cardW - cornerXmax, cardH - cornerYmax], [cardW - cornerXmin, cardH - cornerYmax],
                        [cardW - cornerXmin, cardH - cornerYmin], [cardW - cornerXmax, cardH - cornerYmin]],
                       dtype=np.float32)
refCorners = np.array([refCornerHL, refCornerLR])

'''
    Utility functions
'''


def display_img(img, polygons=[], channels="bgr", size=9):
    """
        Function to display an inline image, and draw optional polygons (bounding boxes, convex hulls) on it.
        Use the param 'channels' to specify the order of the channels ("bgr" for an image coming from OpenCV world)
    """
    if not isinstance(polygons, list):
        polygons = [polygons]
    if channels == "bgr":  # bgr (cv2 image)
        nb_channels = img.shape[2]
        if nb_channels == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.set_facecolor((0, 0, 0))
    ax.imshow(img)
    for polygon in polygons:
        # An polygon has either shape (n,2),
        # either (n,1,2) if it is a cv2 contour (like convex hull).
        # In the latter case, reshape in (n,2)
        if len(polygon.shape) == 3:
            polygon = polygon.reshape(-1, 2)
        patch = patches.Polygon(
            polygon,
            linewidth=1,
            edgecolor='g',
            facecolor='none')
        ax.add_patch(patch)


def generate_filename(dirname, suffixes, prefix=""):
    """
        Function that returns a filename or a list of filenames in directory 'dirname'
        that does not exist yet. If 'suffixes' is a list, one filename per suffix in 'suffixes':
        filename = dirname + "/" + prefix + random number + "." + suffix
        Same random number for all the file name
        Ex:
        > generate_filename("dir","jpg", prefix="prefix")
        'dir/prefix408290659.jpg'
        > generate_filename("dir",["jpg","xml"])
        ['dir/877739594.jpg', 'dir/877739594.xml']
    """
    if not isinstance(suffixes, list):
        suffixes = [suffixes]

    suffixes = [p if p[0] == '.' else '.' + p for p in suffixes]

    while True:
        bname = "%09d" % random.randint(0, 999999999)
        fnames = []
        for suffix in suffixes:
            fname = os.path.join(dirname, prefix + bname + suffix)
            if not os.path.isfile(fname):
                fnames.append(fname)

        if len(fnames) == len(suffixes):
            break

    if len(fnames) == 1:
        return fnames[0]
    else:
        return fnames


class Backgrounds:
    def __init__(self, backgrounds_pck_fn=backgrounds_pck_fn):
        self._images = pickle.load(open(backgrounds_pck_fn, 'rb'))
        self._nb_images = len(self._images)
        # print("Number of images loaded :", self._nb_images)

    def get_random(self, display=False):
        bg = self._images[random.randint(0, self._nb_images - 1)]
        if display:
            plt.imshow(bg)
        return bg


class Noise:
    def __init__(self, random_pck_fn=random_pck_fn):
        self._images = pickle.load(open(random_pck_fn, 'rb'))
        self._nb_images = len(self._images)
        # print("Number of images loaded :", self._nb_images)

    def get_random(self, display=False):
        noise_img = self._images[random.randint(0, self._nb_images - 1)]
        number_of_obj = "noise"
        if display:
            plt.imshow(noise_img)
        return noise_img, number_of_obj


class Cards:
    def __init__(self, cards_pck_fn=cards_pck_fn):
        self._cards = pickle.load(open(cards_pck_fn, 'rb'))
        # self._cards is a dictionary where keys are card names (ex:'Kc') and
        # values are lists of (img,hullHL,hullLR)
        self._nb_cards_by_value = {k: len(self._cards[k]) for k in self._cards}
        # print("Number of cards loaded per name :", self._nb_cards_by_value)

    def get_random(self, card_name=None, display=False):
        if card_name is None:
            card_name = random.choice(list(self._cards.keys()))
        card, hull1, hull2 = self._cards[card_name][random.randint(
            0, self._nb_cards_by_value[card_name] - 1)]
        if display:
            if display:
                display_img(card, [hull1, hull2], "rgb")
        card_name = "card_" + str(card_name)
        return card, card_name, hull1, hull2


class Cash:
    def __init__(self, cash_pck_fn=cash_pck_fn):
        self._images = pickle.load(open(cash_pck_fn, 'rb'))
        self._nb_images = len(self._images)
        # print("Number of images loaded :", self._nb_images)

    def get_random(self, cash_value=None, display=False):
        if cash_value is None:
            cash_value = random.choice(list(self._images.keys()))
        cash = self._images[cash_value][0]
        cash_value = "cash_" + str(cash_value)
        return cash, cash_value


def augment(img, list_kps, seq, restart=True):
    """
        Apply augmentation 'seq' to image 'img' and keypoints 'list_kps'
        If restart is False, the augmentation has been made deterministic outside the function
    """
    # Make sequence deterministic
    while True:
        if restart:
            myseq = seq.to_deterministic()
        else:
            myseq = seq
        # Augment image and keypoints
        img_aug = myseq.augment_images([img])[0]
        list_kps_aug = myseq.augment_keypoints([list_kps])[0]
        valid = True
        # Check the card bounding box stays inside the image
        if valid:
            break
        elif not restart:
            img_aug = None
            break

    return img_aug, list_kps_aug


def calculate_new_polygon(polygon, polygon2):
    polygon_cropped = (polygon.difference(polygon2)
                       ).difference(polygon2)

    if polygon_cropped.geom_type == 'MultiPolygon':
        if polygon_cropped[0].area > polygon_cropped[1].area:
            polygon_cropped = polygon_cropped[0]
        else:
            polygon_cropped = polygon_cropped[1]
    polygon = polygon_cropped
    px, py = polygon_cropped.exterior.coords.xy
    cropped_polygon = []
    for ii in range(0, len(px)):
        point = (px[ii], py[ii])
        cropped_polygon.append(point)

    return polygon, polygon2, cropped_polygon


class CardScene:
    """
        Create a scene with a specified number of card and cash note
    """

    def __init__(self, bg, cards_instance, cash_instance, noise_instance, number_of_card, number_of_cash,
                 number_of_noisy_obj):
        self.cards = cards_instance
        self.cash = cash_instance
        self.noise = noise_instance
        self.number_of_card = number_of_card
        self.number_of_cash = number_of_cash
        self.number_of_noise = number_of_noisy_obj
        self.createNCardsScene(bg, self.number_of_card, self.number_of_cash,
                               self.number_of_noise)

    def createNCardsScene(self, bg, number_of_card, number_of_cash, number_of_noisy_obj):

        imgH = np.random.randint(1500, 3200)
        imgW = int((imgH / 9) * 16)

        img_aug = []
        kp_aug = []
        polygons = []

        self.object_class = []
        self.point_list = []
        self.point_list_cash = []
        number_object = number_of_card + number_of_cash + number_of_noisy_obj

        for i in range(0, self.number_of_noise):
            image = np.zeros((imgH, imgW, 4), dtype=np.uint8)
            noise_image, number_of_obj = self.noise.get_random()

            b_channel, g_channel, r_channel = cv2.split(noise_image)

            alpha_channel = np.ones(b_channel.shape,
                                    dtype=b_channel.dtype) * 50  # creating a dummy alpha channel image.

            noise_image = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
            self.object_class.append(number_of_obj)
            noiseH = noise_image.shape[0]
            noiseW = noise_image.shape[1]
            good_offset = False
            while good_offset == False:
                try:
                    y_offset = np.random.randint(2 * noiseH, imgH - noiseH)
                    x_offset = np.random.randint(2 * noiseW, imgW - noiseW)
                    good_offset = True
                except:
                    pass

            # generating card keypoints based on offset values for augmentation
            cardKP = ia.KeypointsOnImage([
                ia.Keypoint(x=x_offset, y=y_offset),
                ia.Keypoint(x=x_offset + noiseW, y=y_offset),
                ia.Keypoint(x=x_offset + noiseW, y=y_offset + noiseH),
                ia.Keypoint(x=x_offset, y=y_offset + noiseH)
            ], shape=(imgH, imgW, 3))

            # factors for agumenting card
            scale_factor = np.random.uniform(0.8, 1)
            rotate_degree = np.random.randint(-90, 90)
            shear_degree = np.random.randint(-45, 45)

            # transform properties for card
            transform = iaa.Sequential([
                iaa.Affine(scale=scale_factor),
                iaa.Affine(rotate=rotate_degree),
                iaa.Affine(shear=shear_degree),
                iaa.GaussianBlur(sigma=(0, 1.5)),
                iaa.GammaContrast(gamma=(0.3, 1.6))
            ])

            # agumenting the card
            image[y_offset:y_offset + noiseH, x_offset:x_offset + noiseW, :] = noise_image

            image_aug, lkps = augment(image, cardKP, transform)
            img_aug.append(image_aug)
            kp_aug.append(lkps)
            # initialize polygon of the agumented card image
            polygon = Polygon(lkps.get_coords_array())
            polygons.append(polygon)
            x, y = polygon.exterior.xy
            self.point_list.append(list(zip(x, y)))

        for i in range(0, self.number_of_card):
            image = np.zeros((imgH, imgW, 4), dtype=np.uint8)
            card_image, card_val1, hulla1, hullb1 = self.cards.get_random()
            self.object_class.append(card_val1)

            y_offset = np.random.randint(2 * cardH, imgH - cardH)
            x_offset = np.random.randint(2 * cardW, imgW - cardW)

            # generating card keypoints based on offset values for augmentation
            cardKP = ia.KeypointsOnImage([
                ia.Keypoint(x=x_offset, y=y_offset),
                ia.Keypoint(x=x_offset + cardW, y=y_offset),
                ia.Keypoint(x=x_offset + cardW, y=y_offset + cardH),
                ia.Keypoint(x=x_offset, y=y_offset + cardH)
            ], shape=(imgH, imgW, 3))

            # factors for agumenting card
            scale_factor = np.random.uniform(0.8, 1)
            rotate_degree = np.random.randint(-90, 90)
            shear_degree = np.random.randint(-45, 45)

            # transform properties for card
            transform = iaa.Sequential([
                iaa.Affine(scale=scale_factor),
                iaa.Affine(rotate=rotate_degree),
                iaa.Affine(shear=shear_degree),
                iaa.GaussianBlur(sigma=(0, 1.5)),
                iaa.GammaContrast(gamma=(0.3, 1.6))
            ])

            # agumenting the card
            image[y_offset:y_offset + cardH,
            x_offset:x_offset + cardW, :] = card_image

            image_aug, lkps = augment(image, cardKP, transform)
            img_aug.append(image_aug)
            kp_aug.append(lkps)
            # initialize polygon of the agumented card image
            polygon = Polygon(lkps.get_coords_array())
            polygons.append(polygon)
            x, y = polygon.exterior.xy
            self.point_list.append(list(zip(x, y)))

        for i in range(0, self.number_of_cash):
            image = np.zeros((imgH, imgW, 4), dtype=np.uint8)
            cash_image, cash_value = self.cash.get_random()
            self.object_class.append(cash_value)
            cashH = cash_image.shape[0]
            cashW = cash_image.shape[1]

            y_offset = np.random.randint(2 * cashH, imgH - 2 * cashH)
            x_offset = np.random.randint(2 * cashW, imgW - 2 * cashW)

            # generating card keypoints based on offset values for augmentation
            cardKP = ia.KeypointsOnImage([
                ia.Keypoint(x=x_offset, y=y_offset),
                ia.Keypoint(x=x_offset + cashW, y=y_offset),
                ia.Keypoint(x=x_offset + cashW, y=y_offset + cashH),
                ia.Keypoint(x=x_offset, y=y_offset + cashH)
            ], shape=(imgH, imgW, 3))

            # factors for agumenting card
            scale_factor = np.random.uniform(0.8, 1)
            rotate_degree = np.random.randint(-90, 90)
            shear_degree = np.random.randint(-45, 45)

            # transform properties for card
            transform = iaa.Sequential([
                iaa.Affine(scale=scale_factor),
                iaa.Affine(rotate=rotate_degree),
                iaa.Affine(shear=shear_degree),
                iaa.GaussianBlur(sigma=(0, 1.5)),
                iaa.GammaContrast(gamma=(0.3, 1.6))
            ])

            # agumenting the card
            image[y_offset:y_offset + cashH, x_offset:x_offset + cashW, :] = cash_image

            image_aug, lkps = augment(image, cardKP, transform)
            img_aug.append(image_aug)
            kp_aug.append(lkps)
            # initialize polygon of the agumented card image
            polygon = Polygon(lkps.get_coords_array())
            polygons.append(polygon)
            x, y = polygon.exterior.xy
            self.point_list.append(list(zip(x, y)))

        # Calculate polygon for each polygon pair
        for i in range(0, number_object):
            for j in range(0, number_object):
                if i == j:
                    break
                if polygons[j].intersects(polygons[i]):
                    polygon, polygon2, cropped_polygon = calculate_new_polygon(polygons[i], polygons[j])
                    polygons[i] = polygon
                    polygons[j] = polygon2
                    self.point_list[i] = cropped_polygon

        # saving list of polygon points
        list_kp_aug = []
        for poly_points in self.point_list:
            for point in poly_points:
                list_kp_aug.append(ia.Keypoint(x=point[0], y=point[1]))
        self.kp_on_image = ia.KeypointsOnImage(
            list_kp_aug, shape=(imgH, imgW, 3))

        self.point_list2 = []
        for poly in self.point_list:
            poly_lst = []
            for point in poly:
                poly_lst.append((int(point[0]), int(point[1])))
            self.point_list2.append(poly_lst)

        self.point_list = self.point_list2
        scaleBg = iaa.Scale({"height": imgH, "width": imgW})

        # Scaling bg image to final size
        self.bg = scaleBg.augment_image(bg)
        self.final = self.bg
        # Masking the data image objects required for scene with background
        for i in range(number_object - 1, -1, -1):
            mask = img_aug[i][:, :, 3]
            mask_stack = np.stack([mask] * 3, -1)
            self.final = np.where(mask_stack, img_aug[i][:, :, 0:3], self.final)

    def res(self):
        return self.final

    def create_xml(self, xml_file, img_file, point_list):
        """
            An internal function to convert annotation data of data image 
            object(self) to an XML file
            :param xml_file:
            :param img_file:
            :param point_list:
            :return:
        """
        annotation = etree.Element("annotation")
        filename = etree.SubElement(annotation, "filename")
        filename.text = os.path.basename(img_file)

        size = etree.SubElement(annotation, "size")
        height = etree.SubElement(size, "height")
        height.text = str(self.final.shape[0])
        width = etree.SubElement(size, "width")
        width.text = str(self.final.shape[1])
        depth = etree.SubElement(size, "depth")
        depth.text = str(self.final.shape[2])

        segmented = etree.SubElement(annotation, "segmented")
        segmented.text = '0'

        shape_type = etree.SubElement(annotation, "shape_type")
        shape_type.text = "POLYGON"

        for j, poly in enumerate(point_list):
            obj_elem = etree.SubElement(annotation, "object")
            polygon = etree.SubElement(obj_elem, "polygon")
            for i, point in enumerate(poly):
                k = etree.SubElement(polygon, "point" + str(i))
                k.text = str(point[0]) + "," + str(point[1])
            name = etree.SubElement(obj_elem, "name")
            name.text = str(self.object_class[j])
            name_sub = etree.SubElement(obj_elem, "name_sub")
            name_sub.text = str(self.object_class[j])
            pose = etree.SubElement(obj_elem, "pose")
            pose.text = "Unspecified"
            truncated = etree.SubElement(obj_elem, "truncated")
            truncated.text = "0"
            difficulty = etree.SubElement(obj_elem, "difficulty")
            difficulty.text = "0"
            instance_id = etree.SubElement(obj_elem, "instance_id")
            instance_id.text = "0"

        result = etree.tostring(annotation, encoding='unicode')

        with open(xml_file, "w") as f:
            f.write(result)

    def write_files(self, save_dir, display=False):
        """
            write the final image and annotation files to given directory
            :param save_dir:
            :param display: false/true
            :return:
        """
        jpg_fn, xml_fn = generate_filename(save_dir, ["jpg", "xml"])
        plt.imsave(jpg_fn, self.final)
        if display:
            print("New image saved in", jpg_fn)
        self.create_xml(xml_fn, jpg_fn, self.point_list)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--number_image', type=int,
                        help='Number of output image to generate', default=1)
    parser.add_argument('--number_card', type=int, choices=range(0, 8),
                        help='Number of card to paste per output image', default=1)
    parser.add_argument('--number_cash', type=int, choices=range(0, 8),
                        help='Number of cash note to paste per output image', default=1)
    parser.add_argument('--random', type=bool,
                        help='Set this flag to true to get random number of objects in each image', default=False)
    parser.add_argument('--number_noisy', type=int, choices=range(0, 8),
                        help='Number of random noisy object images to paste per output image', default=0)
    parser.add_argument('--save_dir', type=str,
                        help='Directory to save images', default="data/scenes/val")

    return parser.parse_args(argv)


def main(args):
    cards = Cards()
    backgrounds = Backgrounds()
    cash = Cash()
    noise = Noise()

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    if args.random is True:
        for i in tqdm(range(args.number_image)):
            bg = backgrounds.get_random()
            number_of_card = np.random.randint(0, 8)
            number_of_cash = np.random.randint(0, 8)
            number_of_obj = np.random.randint(0, 8)
            newimg = CardScene(bg, cards, cash, noise, number_of_obj, number_of_card, number_of_cash)
            newimg.write_files(args.save_dir)
    else:
        for i in tqdm(range(args.number_image)):
            bg = backgrounds.get_random()
            newimg = CardScene(bg, cards, cash, noise, args.number_noisy, args.number_card, args.number_cash)
            newimg.write_files(args.save_dir)


if __name__ == '__main__':
    start_time_for_program_execution = time.time()
    main(parse_arguments(sys.argv[1:]))
    # Time required to execute program
    print('Execution time:  {0:0.1f} seconds'.format(time.time() - start_time_for_program_execution))