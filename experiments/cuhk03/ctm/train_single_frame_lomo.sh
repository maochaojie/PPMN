DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

cd ../../../
export PYTHONPATH=./experiments_py_lomo/cuhk03/:$PYTHONPATH

caffe/build/tools/caffe train -solver experiments_py_lomo/cuhk03/lomo/solver_lomo.prototxt -gpu $1 \
-snapshot models/lomo/cuhk03/lomo/30000_set02_iter_10000.solverstate
