#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#include "third_party/nccl/nccl.h"
// Run with:
// nccl-abort-hang --total_tasks=2 --port=1234 --id_filename=nccl_id.txt --fail_rank=1 --fail_on_iteration=2 --task=0 &
// nccl-abort-hang --total_tasks=2 --port=1234 --id_filename=nccl_id.txt --fail_rank=1 --fail_on_iteration=2 --task=1
ABSL_FLAG(int, port, -1, "Port");
ABSL_FLAG(int, task, -1, "Task id");
ABSL_FLAG(int, total_tasks, -1, "Total tasks");
ABSL_FLAG(std::string, id_filename, "", "File for generated unique NCCL id ");
ABSL_FLAG(int, fail_rank, -1, "Rank to fail");
ABSL_FLAG(int, fail_on_iteration, -1, "Iteration to fail on");
#define RETURN_IF_ERROR(x) \
  do {                     \
    if (!x.ok()) {         \
      return x;            \
    }                      \
  } while (false);
#define ASSIGN_OR_RETURN(var, e) \
  do {                           \
    if (!e.ok()) {               \
      return e.status();         \
    }                            \
    var = *e;                    \
  } while (false);
// Returns a pretty printed, human readable version of the bytes in the provided
// string, similar to the output of something like hexdump.
std::string PrettyBytes(absl::string_view s) {
  std::vector<std::string> bytes(s.size());
  for (int i = 0; i < s.size(); ++i) {
    bytes[i] = absl::StrFormat("%02x", s[i]);
  }
  return absl::StrJoin(bytes, " ");
}
// Persists the provided ncclUniqueId in the provided file.
absl::Status PersistId(absl::string_view filename, ncclUniqueId id) {
  absl::string_view id_view(id.internal, NCCL_UNIQUE_ID_BYTES);
  std::ofstream ofs;
  ofs.open(std::string(filename), std::ofstream::out);
  ofs << id_view;
  ofs.close();
  return absl::OkStatus();
}
// Periodically polls the the provided file for a serialized ncclUniqueId,
// returning it when it exists.
absl::StatusOr<ncclUniqueId> PollId(absl::string_view filename) {
  std::ifstream stream;
  for (int i = 0; i < 30; i++) {
    if (i > 0) {
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(1000ms);
    }
    stream.open(std::string(filename));
    if (!stream.good()) continue;
    std::stringstream buffer;
    buffer << stream.rdbuf();
    if (!stream.good()) continue;
    if (buffer.str().size() != NCCL_UNIQUE_ID_BYTES) {
      return absl::InternalError("Size mismatch");
    }
    ncclUniqueId id;
    memcpy(id.internal, buffer.str().c_str(), buffer.str().size());
    return id;
  }
  return absl::InternalError("Could not get nccl id from the file");
}
// Converts a NCCL status to an absl status.
absl::Status ToStatus(ncclResult_t result) {
  if (result == ncclSuccess || result == ncclInProgress) {
    return absl::OkStatus();
  }
  return absl::InternalError(
      absl::StrFormat("NCCL operation failed: %s. Last NCCL warning(error) log "
                      "entry (may be unrelated) '%s'.",
                      ncclGetErrorString(result), ncclGetLastError(nullptr)));
}
// Converts a CUDA status to an absl status.
absl::Status ToStatus(cudaError_t err) {
  if (err == cudaSuccess) {
    return absl::OkStatus();
  }
  return absl::InternalError(absl::StrFormat(
      "CUDA operation failed: %s. Last NCCL warning(error) log "
      "entry (may be unrelated) '%s'.",
      cudaGetErrorString(err), cudaGetErrorString(cudaGetLastError())));
}
// Polls the communicator until it is no longer in progress or is aborted.
absl::Status PollUntilDone(ncclComm_t comm, const std::atomic_bool& abort) {
  auto poll = [](ncclComm_t comm,
                 const std::atomic_bool& abort) -> absl::Status {
    ncclResult_t state = ncclInProgress;
    while (state == ncclInProgress) {
      if (abort) {
        return absl::CancelledError("Aborted");
      }
      RETURN_IF_ERROR(ToStatus(ncclCommGetAsyncError(comm, &state)));
    }
    return ToStatus(state);
  };
  absl::Time start = absl::Now();
  absl::Status s = poll(comm, abort);
  absl::Time stop = absl::Now();
  LOG(INFO) << "!!! PollUntilDone polled for " << (stop - start) << ": " << s;
  return s;
}
absl::Status Main() {
  // Parse flags.
  const int rank = absl::GetFlag(FLAGS_task);
  const int n = absl::GetFlag(FLAGS_total_tasks);
  const std::string id_filename = absl::GetFlag(FLAGS_id_filename);
  const int fail_rank = absl::GetFlag(FLAGS_fail_rank);
  const int fail_on_iteration = absl::GetFlag(FLAGS_fail_on_iteration);
  CHECK_GE(rank, 0);
  CHECK_LT(rank, n);
  LOG(INFO) << "!!! Rank " << rank << " / " << n << " starting...";
  // Generate and propagate a unique NCCL id.
  ncclUniqueId id;
  if (rank == 0) {
    RETURN_IF_ERROR(ToStatus(ncclGetUniqueId(&id)));
    LOG(INFO) << "!!! Persisting NCCL unique id to " << id_filename;
    RETURN_IF_ERROR(PersistId(id_filename, id));
  } else {
    LOG(INFO) << "!!! Polling for NCCL unique id in " << id_filename;
    ASSIGN_OR_RETURN(id, PollId(id_filename));
  }
  LOG(INFO) << "!!! NCCL UNIQUE ID = "
            << PrettyBytes(
                   absl::string_view(id.internal, NCCL_UNIQUE_ID_BYTES));
  cudaSetDevice(rank);
  cudaStream_t stream;
  RETURN_IF_ERROR(ToStatus(cudaStreamCreate(&stream)));

  // Create non-blocking communicators.
  std::atomic_bool abort = false;
  ncclComm_t comm;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = 0;
  LOG(INFO) << "!!! Creating communicator...";
  RETURN_IF_ERROR(
      ToStatus(ncclCommInitRankConfig(&comm, n, id, rank, &config)));
  RETURN_IF_ERROR(PollUntilDone(comm, abort));
  LOG(INFO) << "!!! Created communicator";
  for (int i = 0; true; i++) {
    LOG(INFO) << "!!! Running iteration " << i << "...";
    if (rank == fail_rank && i == fail_on_iteration) {
      LOG(ERROR) << "!!! FAILING. GOODBYE";
      return absl::OkStatus();
    }
    // Allocate and initialize buffers.
    // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/data.html
    int32_t host_input[1] = {i};
    int32_t* input = nullptr;
    int32_t* output = nullptr;
    LOG(INFO) << "!!! Allocating buffers...";
    RETURN_IF_ERROR(ToStatus(cudaMalloc(&input, sizeof(int32_t))));
    RETURN_IF_ERROR(ToStatus(cudaMalloc(&output, sizeof(int32_t))));
    RETURN_IF_ERROR(ToStatus(cudaMemcpy(input, host_input, sizeof(int32_t),
                                        cudaMemcpyHostToDevice)));
    RETURN_IF_ERROR(ToStatus(cudaMemset(output, 0, sizeof(int32_t))));
    LOG(INFO) << "!!! Allocated buffers";
    // Perform AllReduce with timeout.
    LOG(INFO) << "!!! Starting AllReduce";
    RETURN_IF_ERROR(ToStatus(
        ncclAllReduce(input, output, 1, ncclInt32, ncclSum, comm, stream)));
    RETURN_IF_ERROR(PollUntilDone(comm, abort));
    LOG(INFO) << "!!! AllReduce started";
    LOG(INFO) << "!!! Synchronizing stream...";
    absl::Status s;
    absl::Notification done;
    std::thread synchronizer([&]() {
      while(cudaStreamQuery(stream) != cudaSuccess) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        LOG(INFO) << "!!!" <<" rank: " << rank <<" stream not ready";
      }
      LOG(INFO) << "!!!" <<" rank: " << rank <<" stream ready";
      done.Notify();
    });
    if (!done.WaitForNotificationWithTimeout(absl::Seconds(30))) {
      LOG(ERROR) << "!!! AllReduce timed out. Aborting";
      RETURN_IF_ERROR(ToStatus(ncclCommAbort(comm)));
      LOG(ERROR) << "!!! Aborted successfully. Waiting for sync";
      synchronizer.join();
      LOG(ERROR) << "!!! Joined synchronizer";
      return absl::OkStatus();
    }
    synchronizer.join();
    RETURN_IF_ERROR(s);
    LOG(INFO) << "!!! Synchronized stream; AllReduce succeeded";
    // Copy AllReduce result buffer from device to host.
    int32_t host_output[1] = {0};
    LOG(INFO) << "!!! Copying to host...";
    RETURN_IF_ERROR(ToStatus(cudaMemcpy(host_output, output, sizeof(int32_t),
                                        cudaMemcpyDeviceToHost)));
    LOG(INFO) << "!!! Copied to host";
    LOG(INFO) << "!!! Reduced output = " << host_output[0] / n;
    // Free buffers.
    LOG(INFO) << "!!! Freeing buffers...";
    RETURN_IF_ERROR(ToStatus(cudaFree(input)));
    RETURN_IF_ERROR(ToStatus(cudaFree(output)));
    LOG(INFO) << "!!! Freed buffers";
    // Sleep.
    LOG(INFO) << "!!! Sleeping for one second...";
    absl::SleepFor(absl::Seconds(1));
    LOG(INFO) << "!!! Slept for one second";
  }
  // Destroy the communicator.
  LOG(INFO) << "!!! Destroying communicator...";
  RETURN_IF_ERROR(ToStatus(ncclCommDestroy(comm)));
  LOG(INFO) << "!!! Destroyed communicator";
}
int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  CHECK_OK(Main());
  return 0;
}