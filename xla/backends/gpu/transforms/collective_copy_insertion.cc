/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/backends/gpu/transforms/collective_copy_insertion.h"

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/side_effect_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "xla/backends/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/service/collective_ops_utils.h"

namespace xla::gpu {

namespace {
bool IsMnnvlEnabledInNccl() {
  const char* mnnvl_enable = std::getenv("NCCL_MNNVL_ENABLE");
  if(!mnnvl_enable) {
    return true;
  }
  return std::stoi(mnnvl_enable);
}
absl::StatusOr<bool> IsRdmaCollective(HloInstruction* instr, const Compiler::CompileOptions* options, int64_t num_visible_devices_per_process, se::GpuComputeCapability gpu_version) {
  int64_t slice_size = options->slice_size;
  if(!IsCollective(instr)) {
    return false;
  }
  TF_ASSIGN_OR_RETURN(
    GPUCommunicationType comm_type,
    CommunicationType(num_visible_devices_per_process, *xla::Cast<HloChannelInstruction>(instr),
                          gpu_version));
  switch(comm_type) {
    case GPUCommunicationType::SINGLE_PARTITION: {
      if(options->gpu_topology.has_value() && slice_size <= options->gpu_topology->num_devices_per_host()) {
        return false;
      }
      return IsMnnvlEnabledInNccl();
    }
    case GPUCommunicationType::MULTI_HOST_WORLD_LEVEL:
    case GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL:
      return true;
    default:
      return false;
  }
  return false;
}

static absl::StatusOr<bool> AddAsyncCopy(HloInstruction* instr) {

  HloComputation* computation = instr->parent();
  for(int64_t op_index = 0; op_index < instr->operand_count(); op_index++) {
    auto operand = instr->mutable_operand(op_index);
    HloInstruction* new_copy = computation->AddInstruction(HloInstruction::CreateUnary(
          operand->shape(), HloOpcode::kCopy, operand));
    FrontendAttributes frontend_attributes;
    (*frontend_attributes.mutable_map())["_xla_stream_annotation"] = "collective";
    new_copy->set_frontend_attributes(frontend_attributes);
    TF_RETURN_IF_ERROR(instr->ReplaceOperandWith(op_index, new_copy));
  }
  VLOG(5) << "Added copy to RDMA op " << instr->ToString();
  return true;
}
}  // namespace

absl::StatusOr<bool> CollectiveCopyInsertion::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (const HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instr : comp->instructions()) {
      bool result = false;
      TF_ASSIGN_OR_RETURN(bool is_rdma_coll, IsRdmaCollective(instr, options_, num_visible_devices_per_process_, gpu_version_));
      if(is_rdma_coll) {
        TF_ASSIGN_OR_RETURN(result, AddAsyncCopy(instr));
      }
      changed |= result;
    }
  }
  return changed;
}

}  // namespace xla::gpu
