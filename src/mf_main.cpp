// Author: Dai Wei (wdai@cs.cmu.edu)
// Date: 2014.04.28

#include "mf_engine.hpp"
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

// All these are required command line inputs
DEFINE_string(data_file, " ", "path to doc file in libsvm format.");
DEFINE_int32(rank, 100, "Factorization rank");
DEFINE_int32(num_iterations, 100, "Number of iterations");
DEFINE_double(init_step_size, 0.5,
    "Initial stochastic gradient descent step size");
DEFINE_string(output_file, " ",
    "Results are in output_file.L and output_file.R.");

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  if (FLAGS_data_file == " ") {
    LOG(FATAL)
      << "usage: need data_file, vocab_file, output_file, and num_partitions";
  }

  mf::MFEngine mf_engine;

  //mf_engine.ReadSparseMatrix(FLAGS_data_file);
  mf_engine.ReadData(FLAGS_data_file);
  mf_engine.Start();

  return 0;
}

