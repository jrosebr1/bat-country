#!/bin/sh

python demo.py --base-model $CAFFE_ROOT/models/bvlc_googlenet \
	--image initial_images/fal_01.jpg \
	--output examples/output_fal_conv2_3x3_01.jpg
python demo.py --base-model $CAFFE_ROOT/models/bvlc_googlenet \
	--image initial_images/fal_02.jpg \
	--output examples/output_fal_conv2_3x3_02.jpg
python demo.py --base-model $CAFFE_ROOT/models/bvlc_googlenet \
	--image initial_images/fal_03.jpg \
	--output examples/output_fal_conv2_3x3_03.jpg
python demo.py --base-model $CAFFE_ROOT/models/bvlc_googlenet \
	--image initial_images/fal_04.jpg \
	--output examples/output_fal_conv2_3x3_04.jpg

python demo.py --base-model $CAFFE_ROOT/models/bvlc_googlenet \
	--image initial_images/fal_01.jpg \
	--output examples/output_fal_inception_3b_5x5_reduce_01.jpg \
	--layer 'inception_3b/5x5_reduce'
python demo.py --base-model $CAFFE_ROOT/models/bvlc_googlenet \
	--image initial_images/fal_02.jpg \
	--output examples/output_fal_inception_3b_5x5_reduce_02.jpg \
	--layer 'inception_3b/5x5_reduce'
python demo.py --base-model $CAFFE_ROOT/models/bvlc_googlenet \
	--image initial_images/fal_03.jpg \
	--output examples/output_fal_inception_3b_5x5_reduce_03.jpg \
	--layer 'inception_3b/5x5_reduce'
python demo.py --base-model $CAFFE_ROOT/models/bvlc_googlenet \
	--image initial_images/fal_04.jpg \
	--output examples/output_fal_inception_3b_5x5_reduce_04.jpg \
	--layer 'inception_3b/5x5_reduce'

python demo.py --base-model $CAFFE_ROOT/models/bvlc_googlenet \
	--image initial_images/fal_01.jpg \
	--output examples/output_fal_inception_4c_output_01.jpg \
	--layer 'inception_4c/output'
python demo.py --base-model $CAFFE_ROOT/models/bvlc_googlenet \
	--image initial_images/fal_02.jpg \
	--output examples/output_fal_inception_4c_output_02.jpg \
	--layer 'inception_4c/output'
python demo.py --base-model $CAFFE_ROOT/models/bvlc_googlenet \
	--image initial_images/fal_03.jpg \
	--output examples/output_fal_inception_4c_output_03.jpg \
	--layer 'inception_4c/output'
python demo.py --base-model $CAFFE_ROOT/models/bvlc_googlenet \
	--image initial_images/fal_04.jpg \
	--output examples/output_fal_inception_4c_output_04.jpg \
	--layer 'inception_4c/output'