// Author: Dai Wei (wdai@cs.cmu.edu)
// Date: 2014.04.28

#include "mf_engine.hpp"
#include "high_resolution_timer.hpp"
#include "context.hpp"
#include <string>
#include <glog/logging.h>
#include <random>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <sstream>

namespace mf {

MFEngine::MFEngine() {
  util::Context& context = util::Context::get_instance();
  rank_ = context.get_int32("rank");
  init_step_size_ = context.get_double("init_step_size");
}

void MFEngine::ReadData(const std::string& file) {
  util::HighResolutionTimer timer;
  char *line = NULL, *ptr = NULL, *endptr = NULL;
  size_t num_bytes;
  FILE *data_stream = fopen(file.c_str(), "r");
  CHECK_NOTNULL(data_stream);
  LOG(INFO) << "Reading from data file " << file;
  int base = 10;

  // Read first line: #-rows/users #-columns/movies
  CHECK_NE(-1, getline(&line, &num_bytes, data_stream));
  N_ = strtol(line, &endptr, base);
  ptr = endptr + 1;   // +1 to skip the space.
  M_ = strtol(ptr, &endptr, base);
  LOG(INFO) << "(#-rows, #-cols) = (" << N_ << ", " << M_ << ")";
  nnz_ = 0;  // number of non-zero entries.
  while (getline(&line, &num_bytes, data_stream) != -1) {
    int row_id = strtol(line, &endptr, base);   // Read the row id.
    ptr = endptr;
    while (*ptr != '\n') {
      // read a word_id:count pair
      int col_id = strtol(ptr, &endptr, base);
      ptr = endptr; // *ptr = colon
      CHECK_EQ(':', *ptr);
      int val = strtol(++ptr, &endptr, base);
      ptr = endptr;
      X_row_idx_.push_back(row_id);
      X_col_idx_.push_back(col_id);
      X_val_.push_back(val);
      ++nnz_;
      while (*ptr == ' ') ++ptr; // goto next non-space char
    }
  }
  LOG(INFO) << "Done reading " << nnz_ << " non-zero entries in "
    << timer.elapsed() << " seconds.";
}

void MFEngine::ReadSparseMatrix(const std::string& inputfile) {
  util::HighResolutionTimer timer;
  X_row_idx_.clear();
  X_col_idx_.clear();
  X_val_.clear();
  N_ = 0;
  M_ = 0;
  std::ifstream inputstream(inputfile.c_str());
  nnz_ = 0;
  while(true) {
    int row, col;
    float val;
    inputstream >> row >> col >> val;
    if (!inputstream) {
      break;
    }
    X_row_idx_.push_back(row);
    X_col_idx_.push_back(col);
    X_val_.push_back(val);
    ++nnz_;
    N_ = row+1 > N_ ? row+1 : N_;
    M_ = col+1 > M_ ? col+1 : M_;
  }
  inputstream.close();
  LOG(INFO) << "Done reading " << nnz_ << " non-zero entries in "
    << timer.elapsed() << " seconds. (N, M) = (" << N_ << ", " << M_ << ")";
}

namespace {

// random generator function:
int myrandom (int i) {
  static std::default_random_engine e;
  return e() % i;
}

}  // anonymous namespace

void MFEngine::Start() {
  std::random_device rd;
  std::mt19937 rng_engine(rd());
  std::uniform_real_distribution<float> zero_to_one_dist(0, 1);

  // Initialize L_ and R_ using N_, M_, and rank_.
  L_ = std::vector<std::vector<float> >(N_, std::vector<float>(rank_));
  R_ = std::vector<std::vector<float> >(M_, std::vector<float>(rank_));
  for (int i = 0; i < N_; ++i) {
    for (int k = 0; k < rank_; ++k) {
      L_[i][k] = (2.0 * zero_to_one_dist(rng_engine) - 1.0);
      //CHECK(L_[i][k] <= 1.0 && L_[i][k] >= -1.) << L_[i][k];
    }
  }
  for (int j = 0; j < M_; ++j) {
    for (int k = 0; k < rank_; ++k) {
      R_[j][k] = (2.0 * zero_to_one_dist(rng_engine) - 1.0);
      //CHECK(R_[j][k] <= 1.0 && R_[j][k] >= -1.) << R_[j][k];
    }
  }

  util::Context& context = util::Context::get_instance();
  int num_iterations = context.get_int32("num_iterations");
  std::vector<int> rand_perm(nnz_);
  for (int i = 0; i < nnz_; ++i) {
    rand_perm[i] = i;
  }
  for (int iter = 0; iter < num_iterations; ++iter) {
    util::HighResolutionTimer iter_timer;
    // permute.
    std::random_shuffle(rand_perm.begin(), rand_perm.end(), myrandom);

    // clear loss
    iter_loss_ = 0.;

    for (int j = 0; j < nnz_; ++j) {
      int idx = rand_perm[j];
      iter_loss_ += DoOneSGD(idx, iter);
    }
    LOG(INFO) << "loss (iter " << iter << "): " << iter_loss_ << ". Time = "
      << iter_timer.elapsed() << " seconds.";
  }

  // Output to file.
  // L_ matrix
  std::string output_file = context.get_string("output_file");
  std::string L_file = output_file + ".L";
  std::ofstream L_stream(L_file.c_str());
  L_stream << PrintL();
  L_stream.close();

  // R_ matrix.
  std::string R_file = output_file + ".R";
  std::ofstream R_stream(R_file.c_str());
  R_stream << PrintL();
  R_stream.close();
}


std::string MFEngine::PrintL() {
  std::stringstream ss;
  for (int i = 0; i < N_; ++i) {
    for (int j = 0; j < rank_; ++j) {
      ss << L_[i][j] << " ";
    }
    ss << std::endl;
  }
  return ss.str();
}

std::string MFEngine::PrintR() {
  std::stringstream ss;
  for (int i = 0; i < M_; ++i) {
    for (int j = 0; j < rank_; ++j) {
      ss << R_[i][j] << " ";
    }
    ss << std::endl;
  }
  return ss.str();
}


float MFEngine::DoOneSGD(int idx, int iter) {
  // Let i = X_row[a], j = X_col[a], and X(i,j) = X_val[a]
  const int i = X_row_idx_[idx];
  const int j = X_col_idx_[idx];
  const float Xij = X_val_[idx];

  std::vector<float>& Li = L_[i];
  std::vector<float>& Rj = R_[j];

  // Compute L(i,:) * R(:,j)
  float LiRj = 0.0;
  for (int k = 0; k < rank_; ++k) {
    LiRj += Li[k] * Rj[k];
    //CHECK_EQ(LiRj, LiRj) << "k = " << k << " Li[k] = " << Li[k] << ", Rj[k] "
    //  << Rj[k];
  }

  // Now update L(i,:) and R(:,j) based on the loss function at X(i,j).
  // The loss function at X(i,j) is ( X(i,j) - L(i,:)*R(:,j) )^2.
  //
  // The gradient w.r.t. L(i,k) is -2 * (X(i,j) - L(i,:)*R(:,j)) * R(k,j).
  // The gradient w.r.t. R(i,k) is -2 * (X(i,j) - L(i,:)*R(:,j)) * L(i,k).
  float step_size = init_step_size_ * std::pow(100.0 + iter, -0.5);
  for (int k = 0; k < rank_; ++k) {
    float gradient = 0.0;
    // Compute update for L(i,k)
    gradient = -2 * (Xij - LiRj) * Rj[k];
    Li[k] -= gradient * step_size;
    //CHECK_EQ(Li[k], Li[k]) << "gradient = " << gradient;
    // Compute update for R(k,j)
    gradient = -2 * (Xij - LiRj) * Li[k];
    Rj[k] -= gradient * step_size;
    //CHECK_EQ(Rj[k], Rj[k]) << "gradient = " << gradient;
  }
  float loss = std::pow(Xij - LiRj, 2);
  //CHECK_EQ(loss, loss) << "Xij = " << Xij << ", LiRj = " << LiRj;
  return loss;
}

}  // namespace mf
