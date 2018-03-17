import numpy as np
import logging
import struct
import random
import os
from scipy import ndimage


class DigitSequence:
    LOG_FILE_NAME = 'digits_sequence.log'
    LOG_DIR = '/tmp/digits_sequence/log'

    def __init__(self, image_idx_file=None, label_idx_file=None):
        """
        Iitialize the class with image idx and label idx file
        :param image_idx_file:
        :param label_idx_file:
        """
        self.__single_image_width = 0
        self.logger = logging.getLogger(__name__)
        self.__set_logger()
        if os.path.exists(image_idx_file) and os.path.exists(label_idx_file):
            self.__image_idx_file = image_idx_file
            self.__label_idx_file = label_idx_file
        else:
            raise ValueError("MNIST Image and label files are not found")

    def __set_logger(self):
        """
        private method for setting logging preferences
        log file will be written in /tmp/
        :return: None
        """
        if not os.path.exists(self.LOG_DIR):
            os.makedirs(self.LOG_DIR)
        f_name = os.path.join(self.LOG_DIR, self.LOG_FILE_NAME)
        fh = logging.FileHandler(f_name, 'w', 'utf-8')
        logging.getLogger().setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(console)

    def __read_images_from_idx(self):
        """
        Read images from the specified files downloaded from MNISt
        :param image_idx_file - string containing the full image file path
        :param label_idx_file - string containing the full label file path

        :return img_array, numpy array containing the images
        :return lbl_array, numpy array containing the labels
        """
        with open(self.__label_idx_file, 'rb') as f_lbl:
            magic, num = struct.unpack(">II", f_lbl.read(8))
            lbl_array = np.fromfile(f_lbl, dtype=np.int8)
        with open(self.__image_idx_file, 'rb') as f_img:
            magic, num, rows, cols = struct.unpack(">IIII", f_img.read(16))
            self.__single_image_width = rows
            img_array = np.fromfile(f_img, dtype=np.uint8).reshape(num, rows, cols)
        return lbl_array, img_array

    def __construct_image_dict(self):
        """
        Construct an image dictionary containing label: list of images
        
        :param None
        :return dictionary , label: list of images
        """
        lbl_array, img_array = self.__read_images_from_idx()
        dict_of_images = {}
        labels_images = list(zip(lbl_array, img_array))
        for i, j in labels_images:
            if i in dict_of_images:
                dict_of_images[i].append(j)
            else:
                dict_of_images[i] = [j]
        return dict_of_images

    def __manage_width(self, digits, spacing_range, width):
        """
        Uses the image width parameter to mange the spacing between digits
        :param digits - list like, sequence of digits
        :param spacing - the original spacing range between digits
        :param width - specified width
        :return new_spacing, an integer representing the new spacing between digits
        
        """
        digit_space = len(digits) * self.__single_image_width
        space_bet_digits = len(digits) - 1
        space_check = round((width - digit_space)/space_bet_digits)
        min_req_width = digit_space + (spacing_range[0] * (len(digits)-1))
        self.logger.debug("Pixels required for digits without spacing" + str(digit_space))
        if width <= min_req_width:
            self.logger.info("Width specified smaller than total digit width, "
                             "should be >= {}, width will be ignored".format( min_req_width))
            spacing = int(np.random.uniform(spacing_range[0], spacing_range[1]))
            self.logger.debug("Spacing :: {}".format(spacing))
        elif space_check in range(spacing_range[0], spacing_range[1]):
            spacing = int(np.random.uniform(spacing_range[0], spacing_range[1]))
            self.logger.info("Width specified sufficient, sequence image width might be greater than w")
            self.logger.info("Total width {}".format(digit_space + space_bet_digits * spacing))
            self.logger.debug("Spacing, Total width :: {}, {}".format(spacing,
                                                                      (digit_space + space_bet_digits * spacing )))
        else:
            # create a spacing that is a uniform distribution between min and width
            new_spacing_max = space_check
            spacing = int(np.random.uniform(spacing_range[0], new_spacing_max))
            self.logger.info("Width specified will accommodate more spacing, new spacing {}".format(spacing))
            self.logger.debug("Spacing to accommodate extra width:: {}".format(spacing))
        return spacing

    def __prechecks(self, digits, spacing_range, image_width):
        """
        Helper method to check the digits against common errors and to
        return the appropriate spacing between digits

        :param digits:
            List like sequence of digits
        :param spacing_range:
            Tuple of minimum and maximum possible spacing range
        :param image_width:
            int - width of the image
        :return:
            list of image numpy arrays specific to those digits
        """
        if not all(isinstance(item, int) for item in digits):
            raise ValueError("Specified digits are not integers")
        if len(list(d for d in digits if int(d/10) == 0)) != len(digits):
            raise ValueError("Only single digits allowed")
        if not all([d >= 0 for d in digits]):
            raise ValueError("Only positive spacing allowed")
        if spacing_range[1] < spacing_range[0]:
            raise ValueError("Max spacing should be >= min spacing")

        d = self.__construct_image_dict()
        spacing = self.__manage_width(digits, spacing_range, image_width)
        images = [random.choice(d[digit]) for digit in digits]
        return images, spacing

    def generate_numbers_sequence(self, digits, spacing_range, image_width):
        """
        Generate an image that contains the sequence of given numbers, spaced
        using a uniform random distribution.

        :param digits:
            A list-like containing the numerical values of the digits from which
            the sequence will be generated (for example [3, 5, 0]).
        :param spacing_range:
            a (minimum, maximum) pair (tuple), representing the min and max spacing
            between digits. Unit should be pixel.
        :param image_width:
            specifies the width of the image in pixels.

        :return numpy array of constructed image
        """
        images, spacing = self.__prechecks(digits, spacing_range, image_width)
        spacing = np.ones([self.__single_image_width, spacing], dtype=float)
        f_images = []
        for index, img in enumerate(images):
            f_images.append(img)
            if index != len(images) -1:
                f_images.append(spacing)
        return np.hstack(f_images)

    def generate_numbers_augment_sequence(self, digits, spacing_range, image_width):
        """
        Generate an image that contains an augmented grid of given numbers, 
        spaced using an uniform distribution. Each digit is flipped 20 degrees
        thrice and added to a column. So given 3 digits you'll receive a 3x3
        grid of digits

        :param digits:
            A list-like containing the numerical values of the digits from which
            the sequence will be generated (for example [3, 5, 0]).
        :param spacing_range:
            a (minimum, maximum) pair (tuple), representing the min and max spacing
            between digits. Unit should be pixel.
        :param image_width:
            specifies the width of the image in pixels.

        :return final numpy array of constructed image
        """
        angle = 20
        images, spacing = self.__prechecks(digits, spacing_range, image_width)
        spacing = np.ones([self.__single_image_width*len(digits), spacing], dtype=float)
        f_images = []
        for index, img in enumerate(images):
            b = np.vstack(np.array([ndimage.interpolation.rotate(input=img,
                                                                 angle=(i+1)*angle,
                                                                 reshape=False) for i in range(len(digits))]))
            f_images.append(b)
            if index != len(images) -1:
                f_images.append(spacing)
        return np.hstack(f_images)
