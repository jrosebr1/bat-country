#!/bin/sh

python demo.py --base-model $CAFFE_ROOT/models/bvlc_googlenet \
	--image initial_images/fear_and_loathing/fal_01.jpg \
	--output examples/output/simple/simple_fal.jpg
python demo.py --base-model $CAFFE_ROOT/models/bvlc_googlenet \
	--image initial_images/the_matrix/matrix_01.jpg \
	--output examples/output/simple/simple_matrix.jpg
python demo.py --base-model $CAFFE_ROOT/models/bvlc_googlenet \
	--image initial_images/jurassic_park/jp_01.jpg \
	--output examples/output/simple/simple_jp.jpg