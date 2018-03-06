DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

cd ../../../
export PYTHONPATH=./experiments_py_lomo/cuhk03/:$PYTHONPATH

caffe/build/tools/caffe train -solver experiments_py_lomo/cuhk03/lomo_assp_fusion/solver.prototxt -gpu $1  \
-weights models/lomo/cuhk03/pretrained/fusion.caffemodel
# -snapshot models/lomo/Viper/lomo_assp_fusion/10000_set02_iter_1000.solverstate
#-snapshot models/lomo/Viper/lomo_assp_fusion/10000_set02_iter_620.solverstate

