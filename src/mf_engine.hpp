// Author: Dai Wei (wdai@cs.cmu.edu)
// Date: 2014.04.28

#pragma once

#include <string>
#include <vector>

namespace mf {

class MFEngine {
public:
  MFEngine();

  void ReadData(const std::string& file);

  // Read sparse data matrix into X_row, X_col and X_val. Each line of the
  // matrix is a whitespace-separated triple (row,col,value), where row>=0 and
  // col>=0.  For example:
  //
  // 0 0 0.5
  // 1 2 1.5
  // 2 1 2.5
  //
  // This specifies a 3x3 matrix with 3 nonzero elements: 0.5 at (0,0), 1.5 at
  // (1,2) and 2.5 at (2,1).
  void ReadSparseMatrix(const std::string& inputfile);

  // Single thread.
  void Start();

  std::string PrintL();
  std::string PrintR();

private:  // private functions
  // Return loss incurred by this observation.
  float DoOneSGD(int idx, int iter);

private:
  // Dimension of X (observed) matrix.
  int N_;  // # of rows (e.g. # of users)
  int M_;  // # of columns (e.g. # of movies)
  int nnz_;   // number of non-zero elements in X.

  // rank of MF.
  int rank_;

  std::vector<int> X_row_idx_;
  std::vector<int> X_col_idx_;
  std::vector<float> X_val_;

  // The L (left) and R (right) matrix.
  std::vector<std::vector<float> > L_;  // of dimension [N_ x rank_]
  std::vector<std::vector<float> > R_;  // of dimension [M_ x rank_]

  float init_step_size_;

  // loss at an iteration.
  float iter_loss_;
};


}  // namespace
