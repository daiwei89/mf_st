
#data_file=datasets/9x9_3blocks
data_file=datasets/netflix.dat
#output_file=results/small_output
output_file=results/netflix_output
rank=100
init_step_size=0.0001
num_iterations=50


GLOG_logtostderr=true \
GLOG_v=-1 \
GLOG_minloglevel=0 \
./bin/mf_main \
 --data_file $data_file \
 --rank $rank \
 --num_iterations $num_iterations \
 --output_file $output_file \
 --init_step_size $init_step_size
