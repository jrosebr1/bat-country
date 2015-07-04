# USAGE
# python demo_bulk.py --base-model $CAFFE_ROOT/models/bvlc_googlenet \
# 	--image initial_images/the_matrix --output examples/output/the_matrix

# import the necessary packages
from __future__ import print_function
from batcountry import BatCountry
from imutils import paths
from PIL import Image
import numpy as np
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--base-model", required=True,
	help="base model path")
ap.add_argument("-i", "--images", required=True,
	help="base path to input directory of images")
ap.add_argument("-o", "--output", required=True,
	help="base path to output directory")
args = ap.parse_args()

# buy the ticket, take the ride
bc = BatCountry(args.base_model)
layers = ("conv2/3x3", "inception_3b/5x5_reduce", "inception_4c/output")

# loop over the input directory of images
for imagePath in paths.list_images(args.images):
	# loop over the layers
	for layer in layers:
		# we can't stop here...
		print("[INFO] processing `{}`".format(imagePath))
		image = bc.dream(np.float32(Image.open(imagePath)), end=layer)

		# write the output image to file
		filename = imagePath[imagePath.rfind("/") + 1:]
		outputPath = "{}/{}_{}".format(args.output, layer.replace("/", "_"), filename)
		result = Image.fromarray(np.uint8(image))
		result.save(outputPath)

# do some cleanup
bc.cleanup()