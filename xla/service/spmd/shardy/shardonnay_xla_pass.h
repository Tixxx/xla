/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_SPMD_SHARDY_SHARDONNAY_XLA_PASS_H_
#define XLA_SERVICE_SPMD_SHARDY_SHARDONNAY_XLA_PASS_H_

#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {
namespace sdy {

// An HloModulePass to run Shardonnay. The pass:
// 1. converts the HLO module into MLIR MHLO and the SDY (Shardonnay) dialect,
// 2. runs Shardonnay passes, including sharding propagation and partitioner,
// 3. converts the MLIR MHLO back to the HLO module.
class ShardonnayXLA : public xla::HloModulePass {
 public:
  explicit ShardonnayXLA(bool runSdyShardingPropagation = true)
      : runSdyShardingPropagation(runSdyShardingPropagation) {}

  absl::string_view name() const override { return "shardonnay-xla"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      xla::HloModule* hloModule,
      const absl::flat_hash_set<absl::string_view>& executionThreads) override;

  void setRunSdyShardingPropagation(bool runSdyShardingPropagation) {
    this->runSdyShardingPropagation = runSdyShardingPropagation;
  }

 private:
  bool runSdyShardingPropagation;
  // TODO. Run other SDY passes with flags.
};

}  // namespace sdy
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SHARDY_SHARDONNAY_XLA_PASS_H_