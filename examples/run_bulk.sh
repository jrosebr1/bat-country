#!/bin/sh

python demo_bulk.py --base-model $CAFFE_ROOT/models/bvlc_googlenet \
	--image initial_images/fear_and_loathing \
	--output examples/output/fear_and_loathing
python demo_bulk.py --base-model $CAFFE_ROOT/models/bvlc_googlenet \
	--image initial_images/the_matrix \
	--output examples/output/the_matrix
python demo_bulk.py --base-model $CAFFE_ROOT/models/bvlc_googlenet \
	--image initial_images/jurassic_park \
	--output examples/output/jurassic_park