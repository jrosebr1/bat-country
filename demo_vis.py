# USAGE
# python demo_vis.py --base-model $CAFFE_ROOT/models/bvlc_googlenet \
#	--image initial_images/fear_and_loathing/fal_01.jpg \
#	--vis examples/output/visualizations

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
ap.add_argument("-v", "--vis", required=True,
	help="path to output directory for visualizations")
args = ap.parse_args()

# we can't stop here...
bc = BatCountry(args.base_model)
(image, visualizations) = bc.dream(np.float32(Image.open(args.image)),
	end=args.layer, visualize=True)
bc.cleanup()

# loop over the visualizations
for (k, vis) in visualizations:
	# write the visualization to file
	outputPath = "{}/{}.jpg".format(args.vis, k)
	result = Image.fromarray(np.uint8(vis))
	result.save(outputPath)