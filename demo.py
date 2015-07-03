# import the necessary packages
from batcountry import BatCountry
from PIL import Image
import numpy as np
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--base-model", required=True, help="base model path")
ap.add_argument("-l", "--layer", type=str, default="conv2/3x3",
	help="layer of CNN to use")
ap.add_argument("-i", "--image", required=True, help="path to base image")
ap.add_argument("-o", "--output", required=True, help="path to output image")
args = ap.parse_args()

# we can't stop here...
bt = BatCountry(args.base_model)
image = bt.dream(np.float32(Image.open(args.image)), end=args.layer)
bt.cleanup()

# write the output image to file
result = Image.fromarray(np.uint8(image))
result.save(args.output)