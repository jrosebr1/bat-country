# USAGE
# python demo_guided.py --base-model $CAFFE_ROOT/models/bvlc_googlenet \
#	--image initial_images/clouds.jpg \
#	--guide-image initial_images/seed_images/starry_night.jpg \
#	--output examples/output/seeded/clouds_and_starry_night.jpg

# import the necessary packages
from batcountry import BatCountry
from PIL import Image
import numpy as np
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--base-model", required=True, help="base model path")
ap.add_argument("-l", "--layer", type=str, default="inception_4c/output",
	help="layer of CNN to use")
ap.add_argument("-i", "--image", required=True, help="path to base image")
ap.add_argument("-g", "--guide-image", required=True, help="path to guide image")
ap.add_argument("-o", "--output", required=True, help="path to output image")
args = ap.parse_args()

# we can't stop here...
bc = BatCountry(args.base_model)
features = bc.prepare_guide(Image.open(args.guide_image), end=args.layer)
image = bc.dream(np.float32(Image.open(args.image)), end=args.layer,
	iter_n=20, objective_fn=BatCountry.guided_objective,
	objective_features=features,)
bc.cleanup()

# write the output image to file
result = Image.fromarray(np.uint8(image))
result.save(args.output)