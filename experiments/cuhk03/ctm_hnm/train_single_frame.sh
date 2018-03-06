DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

cd ../../../
export PYTHONPATH=./experiments_py_lomo/cuhk03/:$PYTHONPATH

caffe/build/tools/caffe train -solver experiments_py_lomo/cuhk03/lomo_hnm/solver.prototxt -gpu $1 \
-snapshot models/lomo/cuhk03/lomo_hnm/15000_set02_v2_all_iter_1000.solverstate
#-weights models/lomo/cuhk03/lomo/30000_set02_iter_30000.caffemodel
