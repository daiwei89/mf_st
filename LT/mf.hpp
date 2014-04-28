#ifndef __MF_MF_HPP__
#define __MF_MF_HPP__

/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

#include <stdint.h>
#include <boost/shared_ptr.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/foreach.hpp>
#include <boost/format.hpp>

#include <cmath>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <utility>
#include <limits>
#include <fstream>

#include "include/lazy-table-module.hpp"
#include "random.hpp"

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using boost::shared_ptr;

enum UnitOfWorkPerClock {
  WORK_UNITS_PER_CLOCK,
  ITERS_PER_CLOCK,
  CLOCKS_PER_ITER
};

namespace mf {

// Performs distributed rank-K matrix factorization using squared-loss
// (Frobenius norm). We partition rows of the NxM data matrix X across instances
// of this solver. N is expected to be > M.
class mf_solver {
 protected:
  // Types
  struct Element {
    uint32_t i;
    uint32_t j;
    double val;
  };

  typedef std::pair<uint32_t, double> uint32_pair_double;

// Member variables

  // Lazytables client
  LazyTableModule& tables_;

  // Data and model variables
  std::vector<Element> elements_;      // Elements that compobj need to process
  std::vector<Element> all_elements_;  // All elements_, needed for compute_obj
  uint32_t nnz_;                       // Number of nonzero elements
  LazyTable& Lm_;                      // NxK left-matrix, row-sparse
  LazyTable& Rm_;                      // KxM right-matrix, column-sparse
  uint32_t t_;                         // Current iteration number (for
                                       // computing step size). Starts at 1.
  // Model parameters
  uint32_t N_, M_;                     // Data matrix dimensions
  uint32_t N_thiscompobj_;             // # of rows allocated to this compobj
  uint32_t K_;                         // Rank of the left/right matrices
  double initial_step_size_;           // Gradient step size at iteration t = 1
  uint32_t num_compobj_;               // # computational objs running in total
  uint32_t compobj_rank_;              // The rank of this compobj
  iter_t staleness_;                   // how much staleness to allow
  uint32_t prefetch_;                  // Prefetching: 0 - disabled,
                                       // 1 - prefetch next iteration,
                                       // 2 - prefetch all iterations in advance
  UnitOfWorkPerClock unit_of_work_per_clock_;
                                       // How work_per_clock_ is specified
  uint32_t work_per_clock_;            // Work per clock
  double project_ball_radius_;         // Project cumulative gradient steps
                                       // in one iteration onto an L2-ball of
                                       // this radius. 0.0 disables.

  // RNG
  Random rng_;                                  // Used only for initialization

  // Temp variables
  uint32_t cur_work_;       // Used when
                            // unit_of_work_per_clock_ == WORK_UNITS_PER_CLOCK
  uint32_t cur_clock_;      // Used when
                            // unit_of_work_per_clock_ == ITERS_PER_CLOCK
  std::vector<std::vector<double> > Lm_cache_;  // Used in conjucntion with
                                                // project_ball_radius_
  std::vector<std::vector<double> > Rm_cache_;  // Used in conjucntion with
                                                // project_ball_radius_

  RowOpVal Lm_update;
  RowOpVal Rm_update;

  // Protected functions

  // Performs gradient descent for data point X(i,j), updating L(i,:) and R(:,j)
  void sgd(uint32_t ele_id) {
    Element& element = elements_[ele_id];
    uint32_t i = element.i;
    uint32_t j = element.j;
    double x = element.val;

    // Read L(i,:) and R(:,j)
    shared_ptr<RowData> Li_ptr = Lm_.read_row(i, staleness_);
    shared_ptr<RowData> Rj_ptr = Rm_.read_row(j, staleness_);
    RowData& Li = *Li_ptr;
    RowData& Rj = *Rj_ptr;

    // Compute L(i,:) * R(:,j)
    double LiRj = 0.0;
    for (uint32_t k = 0; k < K_; ++k) {
      LiRj += Li[k] * Rj[k];
    }

    // Update L(i,:) and R(:,j) based on the loss function at X(i,j).
    //
    // The loss function at X(i,j) is ( X(i,j) - L(i,:)*R(:,j) )^2.
    //
    // The gradient w.r.t. L(i,k) is -2*X(i,j)R(k,j) + 2*L(i,:)*R(:,j)*R(k,j).
    // The gradient w.r.t. R(k,j) is -2*X(i,j)L(i,k) + 2*L(i,:)*R(:,j)*L(i,k).

    // Compute step size
    double effective_iters = 0;
    if (unit_of_work_per_clock_ == WORK_UNITS_PER_CLOCK) {
      effective_iters = t_ * (static_cast<double>(work_per_clock_)/nnz_);
    } else if (unit_of_work_per_clock_ == ITERS_PER_CLOCK) {
      effective_iters = t_ * static_cast<double>(work_per_clock_);
    } else if (unit_of_work_per_clock_ == CLOCKS_PER_ITER) {
      effective_iters = t_ / static_cast<double>(work_per_clock_);
    }
    double step_size =
      initial_step_size_ * pow(100.0 + effective_iters, -0.5);
    double adjusted_step_size =
      step_size * nnz_/static_cast<double>(num_compobj_);

    for (uint32_t k = 0; k < K_; ++k) {
      double gradient = 0.0;
      double update = 0.0;

      // Compute raw gradient wrt L(i,k)
      gradient = (-2 * x * Rj[k]) + (2 * LiRj * Rj[k]);
      // Update L(i,k) with the appropriate step size
      update = -gradient * adjusted_step_size;
      Lm_update[k] = update;

      // Compute raw gradient wrt R(k,j)
      gradient = (-2 * x * Li[k]) + (2 * LiRj * Li[k]);
      // Update R(k,j) with the appropriate step size
      update = -gradient * adjusted_step_size;
      Rm_update[k] = update;
    }
    Lm_.inc(i, Lm_update);
    Rm_.inc(j, Rm_update);
  }

  // Calculates the loss function at data point X(i,j), which is
  // ( X(i,j) - L(i,:)*R(:,j) )^2.
  double calc_obj(Element &element, iter_t obj_staleness = 0) {
    uint32_t i = element.i;
    uint32_t j = element.j;
    double x = element.val;

    // Read L(i,:) and R(:,j)
    shared_ptr<RowData> Rj_ptr = Rm_.read_row(j, obj_staleness);
    shared_ptr<RowData> Li_ptr = Lm_.read_row(i, obj_staleness);
    RowData& Rj = *Rj_ptr;
    RowData& Li = *Li_ptr;

    // Compute L(i,:) * R(:,j)
    double LiRj = 0.0;
    for (uint32_t k = 0; k < K_; ++k) {
      LiRj += Li[k] * Rj[k];
    }

    // Compute and return loss
    return pow(x - LiRj, 2);
  }

 public:
  // Constructor
  mf_solver(uint32_t K_i, double initial_step_size_i,
            uint32_t num_compobj_i, uint32_t compobj_rank_i,
            iter_t staleness_i, uint32_t rng_seed_i, uint32_t prefetch_i,
            UnitOfWorkPerClock unit_of_work_per_clock_i,
            uint32_t work_per_clock_i,
            double project_ball_radius_i,
            LazyTableModule& tables_i, LazyTable& Lm_i, LazyTable& Rm_i)
    :  // LazyTableModule
    tables_(tables_i),
    Lm_(Lm_i), Rm_(Rm_i),
    // Model parameters
    t_(1), K_(K_i),
    // Solver parameters
    initial_step_size_(initial_step_size_i),
    num_compobj_(num_compobj_i), compobj_rank_(compobj_rank_i),
    staleness_(staleness_i), prefetch_(prefetch_i),
    unit_of_work_per_clock_(unit_of_work_per_clock_i),
    work_per_clock_(work_per_clock_i),
    project_ball_radius_(project_ball_radius_i),
    // RNG
    rng_(rng_seed_i) {
    if (project_ball_radius_ < 0.0) {
      project_ball_radius_ = 0.0;
    }
  }

  void read_partbin(std::string inputfile_prefix) {
    std::string inputfile =
      (boost::format("%s.partbin/%i") % inputfile_prefix % compobj_rank_).str();
    std::ifstream inputstream(inputfile.c_str(), std::ios::binary);

    inputstream.read(reinterpret_cast<char *>(&N_), sizeof(N_));
    inputstream.read(reinterpret_cast<char *>(&M_), sizeof(M_));
    inputstream.read(reinterpret_cast<char *>(&nnz_), sizeof(uint32_t));
    inputstream.read(reinterpret_cast<char *>(&N_thiscompobj_),
                     sizeof(uint32_t));
    elements_.resize(N_thiscompobj_);
    inputstream.read(reinterpret_cast<char *>((elements_.data())),
                     sizeof(Element) * N_thiscompobj_);
    inputstream.close();
  }

  void read_sparse_matrix(std::string inputfile, uint32_t load_all_data) {
    std::string line;

    // First round, find the number of nonzero elements
    std::ifstream inputstream(inputfile.c_str());
    getline(inputstream, line);
    std::istringstream ls(line);
    ls >> N_ >> M_;

    for (uint32_t i = 0; i < N_; ++i) {
      getline(inputstream, line);
      std::istringstream linestream(line);
      uint32_t num_el;
      linestream >> num_el;
      nnz_ += num_el;
    }
    inputstream.close();

    // Determine how many elements and the element offset this solver instance
    // should get
    uint32_t div = nnz_ / num_compobj_;
    uint32_t res = nnz_ % num_compobj_;
    N_thiscompobj_ = div + (res > compobj_rank_ ? 1 : 0);
    uint32_t offset = div * compobj_rank_
                      + (res > compobj_rank_ ? compobj_rank_ : res);

    // Second round, read data
    inputstream.open(inputfile.c_str());
    elements_.resize(N_thiscompobj_);
    // If load_all_data > 0, then we store all rows of X (this is required to
    // compute the objective function, even though only elements "belonging" to
    // this compobj are actually needed for gradient descent).
    if (load_all_data) {
      all_elements_.resize(nnz_);
    }
    uint32_t ele_id = 0;
    uint32_t all_ele_id = 0;
    getline(inputstream, line);   // Skip the line for N_ and M_

    // Read data
    for (uint32_t i = 0; i < N_; ++i) {
      getline(inputstream, line);
      std::istringstream linestream(line);
      uint32_t num_el;
      linestream >> num_el;

      if (load_all_data
          || (ele_id < elements_.size() && all_ele_id + num_el > offset)) {
        // Has elements to be loaded in this line
        for (uint32_t j = 0; j < num_el; j ++) {
          uint32_t ele_i = i;
          uint32_t ele_j;
          double ele_val;

          std::string elpair;
          linestream >> elpair;
          uint32_t colonpos = elpair.find(":");

          std::istringstream tmpstream(elpair.substr(0, colonpos));
          tmpstream >> ele_j;
          std::istringstream tmpstream2(elpair.substr(colonpos+1));
          tmpstream2 >> ele_val;

          if (load_all_data) {
            all_elements_[all_ele_id].i = ele_i;
            all_elements_[all_ele_id].j = ele_j;
            all_elements_[all_ele_id].val = ele_val;
          }
          // all_ele_id is always increased no matter whether data is loaded
          all_ele_id++;

          if (ele_id < elements_.size() && all_ele_id > offset) {
            elements_[ele_id].i = ele_i;
            elements_[ele_id].j = ele_j;
            elements_[ele_id].val = ele_val;
            ele_id++;
            if (ele_id == elements_.size()) {
              // Check data loading
              if (all_ele_id != offset + elements_.size()) {
                cerr << "compobj_rank_ = " << compobj_rank_
                     << " div = " << div
                     << " res = " << res
                     << " offset = " << offset
                     << " all_ele_id = " << all_ele_id
                     << " elements_.size() = " << elements_.size()
                     << endl;
                assert(all_ele_id == offset + elements_.size());
              }
            }
          }
        }
      } else {
        // all_ele_id is always increased no matter whether data is loaded
        all_ele_id += num_el;
      }
    }

    // Check data loading
    if (all_ele_id != nnz_) {
      cerr << "all_ele_id = " << all_ele_id
           << " nnz_ = " << nnz_
           << endl;
      assert(all_ele_id == nnz_);
    }

    inputstream.close();

    // std::string outputfile =
        // (boost::format("%s.partbin/%i") % inputfile % compobj_rank_).str();
    // std::ofstream outputstream(
        // outputfile.c_str(), std::ios::out | std::ios::binary);
    // outputstream.write((char *)&N_, sizeof(N_));
    // outputstream.write((char *)&M_, sizeof(M_));
    // outputstream.write((char *)&nnz_, sizeof(nnz_));
    // outputstream.write((char *)&N_thiscompobj_, sizeof(N_thiscompobj_));
    // outputstream.write(
        // (char *)(elements_.data()), sizeof(Element) * N_thiscompobj_);
    // outputstream.close();
  }

  // Reads sparse matrix data, and then
  // randomly initializes elements_ of L, R, in the range [-1,1). This should be
  // called by ALL instances of mf_solver, and only ONCE per instance.
  //
  // For L, we only initialize rows belonging to this instance.
  //
  // For R, we initialize all columns in [-1/num_compobj_,1/num_compobj).
  // Thus, once Lazytables has propagated the initialization, the effective
  // range is still [-1,1).
  void initialize() {
#if defined(VECTOR_OPLOG)
    Lm_update.resize(K_);
    Rm_update.resize(K_);
#endif
    std::set<uint32_t> i_set;
    std::set<uint32_t> j_set;
    for (uint32_t i = 0; i < elements_.size(); i ++) {
      i_set.insert(elements_[i].i);
      j_set.insert(elements_[i].j);
      assert(elements_[i].i < N_);
      assert(elements_[i].j < M_);
    }
    BOOST_FOREACH(uint32_t i, i_set) {
      for (uint32_t k = 0; k < K_; ++k) {
        double value = (2.0 * rng_.rand_r() - 1.0) / num_compobj_;
        Lm_update[k] = value;
      }
      Lm_.inc(i, Lm_update, false /* no stat */);
    }
    BOOST_FOREACH(uint32_t j, j_set) {
      for (uint32_t k = 0; k < K_; ++k) {
        double value = (2.0 * rng_.rand_r() - 1.0) / num_compobj_;
        Rm_update[k] = value;
      }
      Rm_.inc(j, Rm_update, false /* no stat */);
    }

    // Reset counters
    cur_work_ = 0;
    cur_clock_ = 0;
  }

  void virtual_access_element(uint32_t i) {
    Element& element = elements_[i];
    Lm_.virtual_read(element.i, staleness_);
    Rm_.virtual_read(element.j, staleness_);
    Lm_.virtual_write(element.i);
    Rm_.virtual_write(element.j);
  }

  void virtual_iteration() {
    // Do a virtual iteration over the whole dataset, so that the LazyTable
    // library can better serve it.

    if (unit_of_work_per_clock_ == WORK_UNITS_PER_CLOCK) {
      // # work units per clock
      // Since the access pattern is not periodic with this setting,
      // auto-prefetching might not be able to work well in this case.
      uint32_t i = 0;
      while (i < elements_.size()) {
        for (uint32_t work_done = 0;
             work_done < work_per_clock_; work_done++) {
          virtual_access_element(i);
          i++;
          if (i >= elements_.size()) {
            break;
          }
          tables_.virtual_clock();
        }
      }
      tables_.finish_virtual_iteration();
    } else if (unit_of_work_per_clock_ == ITERS_PER_CLOCK) {
      // # iterations per clock
      for (uint32_t i = 0; i < elements_.size(); i++) {
        virtual_access_element(i);
      }
      tables_.finish_virtual_iteration();
      // No virtual clock command in this case, because we don't want the
      // client library to record redundant access patterns.
      // The con of this approach is that we don't know its work per clock.
    } else if (unit_of_work_per_clock_ == CLOCKS_PER_ITER) {
      // # clocks per iteration
      for (uint32_t clock = 0; clock < work_per_clock_; clock++) {
        uint32_t begin = elements_.size() * clock / work_per_clock_;
        uint32_t end = elements_.size() * (clock + 1) / work_per_clock_;
        for (uint32_t i = begin; i < end; i++) {
          virtual_access_element(i);
        }
        tables_.virtual_clock();
      }
      tables_.finish_virtual_iteration();
    }
  }

  void refresh_cache() {
    std::set<uint32_t> i_set;
    std::set<uint32_t> j_set;
    for (uint32_t i = 0; i < elements_.size(); i ++) {
      Element& element = elements_[i];
      if (!i_set.count(element.i)) {
        Lm_.refresh_row(element.i, 0);
        i_set.insert(element.i);
      }
      if (!j_set.count(element.j)) {
        Rm_.refresh_row(element.j, 0);
        j_set.insert(element.j);
      }
    }

    BOOST_FOREACH(uint32_t i, i_set) {
      Lm_.read_row(i, 0, false /* no stat */);
    }
    BOOST_FOREACH(uint32_t j, j_set) {
      Rm_.read_row(j, 0, false /* no stat */);
    }
  }

  void turn_prefetch(bool onoff) {
    tables_.turn_prefetch(onoff);
  }

  // Invokes the MF solver for the next wpi_ rows of Xm belonging
  // to this instance, performing gradient descent exactly once on every
  // non-missing element of X in those rows.
  void solve() {
    if (unit_of_work_per_clock_ == WORK_UNITS_PER_CLOCK) {
      // # work units per clock
      for (uint32_t work_done = 0; work_done < work_per_clock_; work_done++) {
        uint32_t i = cur_work_;
        sgd(i);
        // Advance work counter
        cur_work_ = (cur_work_ + 1) % N_thiscompobj_;
      }
    } else if (unit_of_work_per_clock_ == ITERS_PER_CLOCK) {
      // # iterations per clock
      for (uint32_t work_done = 0; work_done < work_per_clock_; work_done++) {
        for (uint32_t i = 0; i < elements_.size(); i++) {
          sgd(i);
        }
      }
    } else if (unit_of_work_per_clock_ == CLOCKS_PER_ITER) {
      // # clocks per iteration
      uint32_t cur_clock_mod = cur_clock_ % work_per_clock_;
      uint32_t begin = elements_.size() * cur_clock_mod / work_per_clock_;
      uint32_t end = elements_.size() * (cur_clock_mod + 1) / work_per_clock_;
      for (uint32_t i = begin; i < end; i++) {
        sgd(i);
      }
      // Advance clock counter
      cur_clock_++;
    }

    // Increase Lazytables and step-size iteration numbers
    ++t_;
    tables_.iterate();
  }

  // Forces an iteration to occur with no work done
  void force_iterate() {
    tables_.iterate();
  }

  // Computes the loss function.
  double calc_obj(iter_t obj_staleness = 0) {
    double obj = 0.0;
    int non_zeros_X = 0;

    // Fetch rows
    for (uint32_t i = 0; i < N_; i ++) {
      Lm_.refresh_row(i, obj_staleness);
    }
    for (uint32_t j = 0; j < M_; j ++) {
      Rm_.refresh_row(j, obj_staleness);
    }

    for (uint32_t i = 0; i < all_elements_.size(); i ++) {
      non_zeros_X++;
      obj += calc_obj(all_elements_[i], obj_staleness);
    }
    std::cout << "non_zeros_X = " << non_zeros_X << std::endl;

    return obj;
  }

  // Outputs the factors L, R to disk
  void output_to_disk(std::string prefix) {
    // L
    std::string Lmfile = prefix + ".L";
    std::ofstream Lm_stream(Lmfile.c_str());
    for (uint32_t i = 0; i < N_; ++i) {
      shared_ptr<RowData> Lmrow_ptr = Lm_.read_row(i, 0);
      RowData& Lmrow = *Lmrow_ptr;
      for (uint32_t k = 0; k < K_; ++k) {
        Lm_stream << Lmrow[k] << " ";
      }
      Lm_stream << "\n";
    }
    Lm_stream.close();
    // R
    std::string Rmfile = prefix + ".R";
    std::ofstream Rm_stream(Rmfile.c_str());
    for (uint32_t j = 0; j < M_; ++j) {
      shared_ptr<RowData> Rmrow_ptr = Rm_.read_row(j, 0);
      RowData& Rmrow = *Rmrow_ptr;
      for (uint32_t k = 0; k < K_; ++k) {
        Rm_stream << Rmrow[k] << " ";
      }
      Rm_stream << "\n";
    }
    Rm_stream.close();
  }
};  // mf_solver



// Worker thread class to execute mf_solver objects
class mf_thread_executer {
  std::string data_file_;
  uint32_t K_;
  double initial_step_size_;
  iter_t staleness_;
  uint32_t rng_seed_;
  LazyTableModule& tables_;
  boost::barrier& sync_barrier_;
  uint32_t num_iterations_;
  uint32_t stat_iter_;
  double time_limit_;
  uint32_t report_obj_;
  uint32_t prefetch_;
  UnitOfWorkPerClock unit_of_work_per_clock_;
  uint32_t work_per_clock_;
  uint32_t load_all_data_;
  double project_ball_radius_;
  uint32_t num_process_, process_rank_;
  uint32_t num_compobj_, compobj_rank_;
  bool write_matrices_;

 public:
  mf_thread_executer(std::string data_file_i,
                     uint32_t K_i, double initial_step_size_i,
                     iter_t staleness_i,
                     uint32_t rng_seed_i, LazyTableModule& tables_i,
                     boost::barrier& sync_barrier_i,
                     uint32_t num_iterations_i, uint32_t stat_iter_i,
                     double time_limit_i,
                     uint32_t report_obj_i, uint32_t prefetch_i,
                     UnitOfWorkPerClock unit_of_work_per_clock_i,
                     uint32_t work_per_clock_i,
                     uint32_t load_all_data_i,
                     double project_ball_radius_i,
                     uint32_t num_process_i, uint32_t process_rank_i,
                     uint32_t num_compobj_i, uint32_t compobj_rank_i,
                     bool write_matrices_i)
    : data_file_(data_file_i),
      K_(K_i), initial_step_size_(initial_step_size_i),
      staleness_(staleness_i), rng_seed_(rng_seed_i), tables_(tables_i),
      sync_barrier_(sync_barrier_i),
      num_iterations_(num_iterations_i), stat_iter_(stat_iter_i),
      time_limit_(time_limit_i),
      report_obj_(report_obj_i), prefetch_(prefetch_i),
      unit_of_work_per_clock_(unit_of_work_per_clock_i),
      work_per_clock_(work_per_clock_i),
      load_all_data_(load_all_data_i),
      project_ball_radius_(project_ball_radius_i),
      num_process_(num_process_i), process_rank_(process_rank_i),
      num_compobj_(num_compobj_i), compobj_rank_(compobj_rank_i),
      write_matrices_(write_matrices_i)
  { }

  void operator()() {
    std::cout << "Compobj " << compobj_rank_
              << ", Process " << process_rank_
              << ": Starting...\n";
    // Am I the "head" compobj/thread on this process?
    bool is_leading = compobj_rank_ < num_process_;

    // Initialize LT
    tables_.thread_start();
    sync_barrier_.wait();
    if (is_leading) {
      MPI_Barrier(MPI_COMM_WORLD);
    }
    sync_barrier_.wait();

    LazyTable Lm = tables_.create_table("Lm_", 0 /* UNKNOWN */, K_);
    LazyTable Rm = tables_.create_table("Rm_", 0 /* UNKNOWN */, K_);

    // Create solver object
    mf_solver solver(K_, initial_step_size_, num_compobj_, compobj_rank_,
                     staleness_, rng_seed_, prefetch_,
                     unit_of_work_per_clock_, work_per_clock_,
                     project_ball_radius_,
                     tables_, Lm, Rm);

    // Force an iteration so clients/servers discover each other
    solver.force_iterate();
    sync_barrier_.wait();
    if (is_leading) {
      MPI_Barrier(MPI_COMM_WORLD);
    }
    sync_barrier_.wait();

    // Read data
    // solver.read_sparse_matrix(data_file_, load_all_data_);
    solver.read_partbin(data_file_);

    // Provide the client library some insights about the access patterns
    double vi_start = MPI_Wtime();
    solver.virtual_iteration();
    double vi_time = MPI_Wtime() - vi_start;

    // Set initial values
    solver.initialize();

    // Push updates
    solver.force_iterate();
    sync_barrier_.wait();
    if (is_leading) {
      MPI_Barrier(MPI_COMM_WORLD);
    }
    sync_barrier_.wait();

    // Refresh cache
    solver.refresh_cache();

    // Start timing
    solver.force_iterate();
    sync_barrier_.wait();
    if (is_leading) {
      MPI_Barrier(MPI_COMM_WORLD);
    }
    sync_barrier_.wait();

    // Run solver
    double start_time = MPI_Wtime();
    double end_time = 0;
    iter_t last_iter = 0;
    for (uint32_t i = 0; i < num_iterations_; ++i) {
      //   Clock 0: let the servers see the client
      //   Clock 1: Set initial data value to the table
      //   Clock 2: Refresh all table rows
      //   ...
      //   Clock N+2: The last application clock

      // Report objective (loss) function if requested.
      // Compobjs take turns to compute it.
      if (report_obj_ > 0 && i % report_obj_ ==
          0 && (i/report_obj_) % num_compobj_ == compobj_rank_) {
        double elapsed_time = MPI_Wtime() - start_time;
        double obj = load_all_data_ == 1 ? solver.calc_obj() : 0;
        std::cout << "(compobj = " << compobj_rank_
                  << ", process = " << process_rank_
                  << ", iteration = " << i+1
                  << ", time = " << elapsed_time
                  << "): loss function = " << obj
                  << "\n";
      }

      solver.solve();
      last_iter++;

      // Stop if time limit reached
      double elapsed_time = MPI_Wtime() - start_time;
      if (time_limit_ > 0.0 && elapsed_time > time_limit_) {
        end_time = elapsed_time + start_time;
        // Spin iterations until we hit the iteration limit.
        // This hack is needed to compute the final objective.
        ++i;
        while (i < num_iterations_) {
          solver.force_iterate();
          ++i;
        }
        break;
      }
    }

    double work_finish_time = MPI_Wtime();

    // Wait until all compobjs in all processes have finished
    sync_barrier_.wait();
    // Am I the "head" compobj/thread on this process?
    if (compobj_rank_ < num_process_) {
      MPI_Barrier(MPI_COMM_WORLD);
    }
    sync_barrier_.wait();

    // Output final loss function
    if (end_time == 0) {
      end_time = MPI_Wtime();
    }

    string json_stats;
    if (is_leading) {
      json_stats = tables_.json_stats();
    }
    if (compobj_rank_ == 0) {
      std::cout << "Solver time = " << end_time - start_time << " seconds\n";
      double obj = load_all_data_ == 1 ? solver.calc_obj() : 0;
      std::cout << "Final loss function = " << obj << "\n";
      std::cout << "RESULTS: { "
                << "\"time\": " << end_time - start_time
                << ", \"wait_other_worker_time\": "
                << end_time - work_finish_time
                << ", \"vi_time\": " << vi_time
                << ", \"data_file\": \"" << data_file_ << "\""
                << ", \"objective\": " << obj
                << ", \"nr_iteration\": " << last_iter
                << ", \"unit_of_work_per_clock\": " << unit_of_work_per_clock_
                << ", \"work_per_clock\": " << work_per_clock_
                << ", \"staleness\": " << staleness_
                << ", \"project_ball_radius\": " << project_ball_radius_
                << ", \"prefetch\": " << prefetch_
                << ", \"stats\": " << json_stats
                << " }" << endl;
    }

    // Triple barrier, just to make sure that Thread 0 can get all of
    // our updates eventually
    sync_barrier_.wait();
    // Am I the "head" compobj/thread on this process?
    if (compobj_rank_ < num_process_) {
      MPI_Barrier(MPI_COMM_WORLD);
    }
    sync_barrier_.wait();

    // Output to disk
    if (write_matrices_ && compobj_rank_ == 0) {
      size_t lastslash = data_file_.find_last_of("/");
      std::string data_file_nopath = lastslash ==
        std::string::npos ? data_file_ : data_file_.substr(lastslash+1);
      solver.output_to_disk(data_file_nopath);
    }

    // Stop LT
    tables_.thread_stop();
  }
};

}  // namespace mf

#endif  // defined __MF_MF_HPP__
