/*
 * Copyright (C) 2013 by Carnegie Mellon University.
 */

#include <mpi.h>
#include <boost/program_options.hpp>
#include <boost/thread/thread.hpp>

#include <iostream>
#include <string>
#include <vector>

#include "include/lazy-table-module.hpp"

#include "mf.hpp"

namespace po = boost::program_options;

using std::string;
using std::vector;
using mf::mf_thread_executer;

void parse_hostfile(std::string hostfile, std::vector<std::string>& hostlist) {
  std::ifstream is(hostfile.c_str());
  std::string line;
  hostlist.clear();
  while (!!getline(is, line)) {
    hostlist.push_back(line);
  }
  is.close();
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int num_processes, process_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

  string hostfile;
  uint32_t threads;
  string datafile;
  uint32_t K, num_iterations, rng_seed;
  iter_t staleness;
  uint32_t stat_iter;
  double time_limit;
  double init_step_size;
  uint32_t report_obj;
  uint32_t prefetch;
  iter_t pf_advance;
  double psafe;
  uint32_t pp_policy;
  uint32_t local_opt;
  uint32_t tcp_base_port;
  uint32_t num_comm_channels;
  string output_dir;
  uint32_t sync_mode;
  iter_t start_clock;
  iter_t log_interval;
  iter_t snapshot_interval;
  uint32_t cache_type;
  uint32_t cache_size;
  uint32_t unit_of_work_per_clock;
  uint32_t work_per_clock;
  uint32_t load_all_data;
  double project_ball_radius;
  bool write_matrices;

  po::options_description desc("Allowed options");
  desc.add_options()
    ("hostfile",
     po::value<string>(&hostfile)->default_value("singlemachine.zmq"),
     "LazyTableModule zmq-formatted hostfile")
    ("threads",
     po::value<uint32_t>(&threads)->default_value(8),
     "Number of threads per process")
    ("data",
     po::value<string>(&datafile),
     "Sparse matrix data file")
    ("K",
     po::value<uint32_t>(&K),
     "Factorization rank")
    ("iters",
     po::value<uint32_t>(&num_iterations),
     "Number of iterations")
    ("stat_iter", po::value<uint32_t>(&stat_iter)->default_value(100),
     "The iteration to report stat")
    ("staleness",
     po::value<iter_t>(&staleness),
     "Bound on data staleness")
    ("init_step_size",
     po::value<double>(&init_step_size)->default_value(1e-7),
     "Initial step size")
    ("rng_seed",
     po::value<uint32_t>(&rng_seed)->default_value(12345),
     "RNG seed")
    ("time_limit",
     po::value<double>(&time_limit)->default_value(0.0),
     "Time limit in seconds (0 or negative disables)")
    ("report_obj",
     po::value<uint32_t>(&report_obj)->default_value(0),
     "Report objective function every X iterations (0 disables)")
    ("prefetch", po::value<uint32_t>(&prefetch)->default_value(0),
     "Prefetching: 0 - disabled, 1 - prefetch only next iteration, "
     "2 - prefetch all iterations in advance")
    ("pf_advance", po::value<iter_t>(&pf_advance)->default_value(0),
     "Prefetch advance")
    ("psafe", po::value<double>(&psafe)->default_value(0.1), "P_safe value")
    ("output_dir", po::value<string>(&output_dir)->default_value("./"),
     "Output directory")
    ("sync_mode", po::value<uint32_t>(&sync_mode)->default_value(0),
     "Synchronization mode: SSP - 0, ASYNC - 1")
    ("log_interval", po::value<iter_t>(&log_interval)->default_value(0),
     "Logging interval")
    ("snapshot_interval",
     po::value<iter_t>(&snapshot_interval)->default_value(0),
     "Snapshot interval")
    ("start_clock", po::value<iter_t>(&start_clock)->default_value(3),
     "The clock we consider as start")
    ("cache_type",
     po::value<uint32_t>(&cache_type)->default_value(0),
     "Cache type")
    ("cache_size", po::value<uint32_t>(&cache_size)->default_value(0),
     "Cache size")
    ("pp_policy", po::value<uint32_t>(&pp_policy)->default_value(3),
     "Parameter Partitioning Policy")
    ("local_opt", po::value<uint32_t>(&local_opt)->default_value(1),
     "Local communication optimization")
    ("num_channels", po::value<uint32_t>(&num_comm_channels)->default_value(1),
     "Number of communication channels")
    ("tcp_base_port", po::value<uint32_t>(&tcp_base_port)->default_value(9090),
     "Basic TCP port")
    ("unit_of_work_per_clock",
     po::value<uint32_t>(&unit_of_work_per_clock)->default_value(0),
     "How work_per_clock is specified:"
     "0: work units per clock"
     "1: iterations per clock"
     "2: clocks per iteration")
    ("work_per_clock",
     po::value<uint32_t>(&work_per_clock),
     "# work units in one clock."
     "It's dependent on the value of unit_of_work_per_clock")
    ("load_all_data",
     po::value<uint32_t>(&load_all_data)->default_value(0),
     "Should each thread load all data? "
     "If 0, objective computations will be invalid.")
    ("project_ball_radius",
     po::value<double>(&project_ball_radius)->default_value(0.0),
     "Project the gradient at each iteration onto an L2-ball of this radius. "
     "0.0 disbales.")
    ("write_matrices",
     po::value<bool>(&write_matrices)->default_value(0),
     "Write factors L, R to disk after execution?");
  po::variables_map options_map;
  po::store(po::parse_command_line(argc, argv, desc), options_map);
  po::notify(options_map);

  // Parse hostfile and start LT client
  vector<string> host_list;
  parse_hostfile(hostfile, host_list);
  LazyTableConfig config;
  config.host_list = host_list;
  config.tcp_base_port = tcp_base_port;
  config.num_comm_channels = num_comm_channels;
  config.output_dir = output_dir;
  config.start_clock = start_clock;
  config.snapshot_interval = snapshot_interval;
  config.log_interval = log_interval;
  config.sync_mode = SyncMode(sync_mode);
  config.prefetch = prefetch;
  config.pf_advance = pf_advance;
  config.psafe = psafe;
  config.pp_policy = pp_policy;
  config.local_opt = local_opt;
  LazyTableModule tables(process_rank, config);

  // Run parallel solvers
  uint32_t num_compobj = num_processes * threads;
  boost::barrier sync_barrier(threads);
  boost::thread_group workers_executer;
  for (uint32_t i = 0; i < threads; ++i) {
    uint32_t compobj_rank = process_rank + i * num_processes;
    mf_thread_executer new_worker(datafile, K, init_step_size,
                                  staleness, rng_seed,
                                  tables, sync_barrier, num_iterations,
                                  stat_iter, time_limit,
                                  report_obj, prefetch,
                                  UnitOfWorkPerClock(unit_of_work_per_clock),
                                  work_per_clock,
                                  load_all_data, project_ball_radius,
                                  num_processes, process_rank,
                                  num_compobj, compobj_rank, write_matrices);
    workers_executer.create_thread(new_worker);
  }
  workers_executer.join_all();

  // Finish and exit
  tables.shutdown();
  MPI_Finalize();
}
