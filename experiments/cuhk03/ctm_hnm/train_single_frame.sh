DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

cd ../../../
export PYTHONPATH=./experiments/cuhk03/:$PYTHONPATH

caffe/build/tools/caffe train -solver experiments/cuhk03/ctm_hnm/solver.prototxt -gpu $1 \
