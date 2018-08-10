
import cv2
import numpy as np

"""
Background subtractor
"""

class BackGroundSubtractor:
    """
    Background substractor using open cv2 lib
    """
    def __init__(self, alpha, threshold, first_image):
        """
        Default constructor
        Args:
          :alpha: The background learning factor, its value should
        be between 0 and 1. The higher the value, the more quickly
        your program learns the changes in the background. Therefore,
        for a static background use a lower value, like 0.001. But if
        your background has moving trees and stuff, use a higher value,
        maybe start with 0.01.
          :threshold: Threshold value used for binary thresholding while
            computing the motion image.
          :first_frame: This is the first image of the alarm
        """
        self.alpha  = alpha
        self.threshold = threshold
        self.back_ground_model = self.denoise(first_image)

    def get_motion_image(self, frame):
        """Apply the background averaging formula:
        NEW_BACKGROUND = CURRENT_FRAME * ALPHA + OLD_BACKGROUND * (1 - APLHA)
        Arg:
          :frame: original image
        Return:
          :thresh_img_gray: Binary motion image based on the background
            subtraction model.
        """
        frame = self.denoise(frame)
        self.back_ground_model =  frame*self.alpha \
                                  + self.back_ground_model * (1-self.alpha)

        # after the previous operation, the dtype of
        # self.backGroundModel will be changed to a float type
        # therefore we do not pass it to cv2.absdiff directly,
        # instead we acquire a copy of it in the uint8 dtype
        # and pass that to absdiff.
        fore_ground = cv2.absdiff(
            self.back_ground_model.astype(np.uint8),
            frame,
        )
        # thresholding operation to get a binary motion image
        _, thresh_img = cv2.threshold(fore_ground, self.threshold, 255,
            cv2.THRESH_BINARY)
        # get single channel motion image
        thresh_img_gray = cv2.cvtColor(thresh_img, cv2.COLOR_BGR2GRAY)

        return thresh_img_gray

    def denoise(self, frame):
        """
        Denoising function to smooth the image before applying the background
        substraction
        Args:
          :frame: original image (RGB)
        Return:
          :frame: modified image (RGB)
        """
        frame = cv2.medianBlur(frame, 5)
        frame = cv2.GaussianBlur(frame, (5,5), 0)
        return frame

