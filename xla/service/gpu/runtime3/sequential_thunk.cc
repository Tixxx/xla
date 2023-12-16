/* Copyright 2017 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/gpu/runtime3/sequential_thunk.h"

#include "xla/status.h"
#include "tsl/platform/errors.h"
#include "tsl/profiler/lib/scoped_annotation.h"

namespace xla {
namespace gpu {

using ::tsl::profiler::ScopedAnnotation;

SequentialThunk::SequentialThunk(ThunkInfo thunk_info, ThunkSequence thunks)
    : Thunk(Kind::kSequential, thunk_info), thunks_(std::move(thunks)) {
  for (auto& thunk : thunks_) {
    // If any thunk runs on a different compute stream,
    // initialize async executor here.
    if (thunk->execution_stream_id() != kMainComputeStreamId) {
      async_ = std::make_unique<AsyncExecutorBase>();
      break;
    }
  }
}

std::string SequentialThunk::ToStringExtra(int indent) const {
  std::string result = "\n";
  absl::StrAppend(&result, thunks().ToString(indent + 1, nullptr));
  return result;
}

absl::Status SequentialThunk::Initialize(const InitializeParams& params) {
  for (auto& thunk : thunks_) {
    TF_RETURN_IF_ERROR(thunk->Initialize(params));
  }
  return absl::OkStatus();
}

absl::Status SequentialThunk::ExecuteOnStream(const ExecuteParams& params) {
  for (const auto& thunk : thunks_) {
    ScopedAnnotation annotation([&] { return thunk->profile_annotation(); });
    if (thunk->wait_on_streams().size() > 0) {
      VLOG(2) << "SequentialThunk waiting for source streams.";
      TF_RETURN_IF_ERROR(async_->Await(params));
    }
    int64_t stream_id = thunk->execution_stream_id();
    VLOG(2) << "SequentialThunk Running thunk on stream: " << stream_id;
    // Run on the target compute stream
    if (stream_id != kMainComputeStreamId) {
      absl::Status status = [&]() {
        return async_->Execute(
            [&](const ExecuteParams& params) {
              return thunk->ExecuteOnStream(params);
            },
            params, stream_id);
      }();
      TF_RETURN_IF_ERROR(status);
    } else {
      TF_RETURN_IF_ERROR(thunk->ExecuteOnStream(params));
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
