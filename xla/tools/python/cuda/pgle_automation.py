
# This is a driver script to sweep through a series of runs with
# different allreduce combiner threshold sizes and return the most optimal
# threshold that leads to the best overlap percentage(the larger the better).
# This script will perform actions in the following sequence:
# 1. Perform profiling run. Note that in this step, if input 'xla_flags' has turned on async collective ops, the script will
#    automatically turn them off to get the accurate collective runtime.
# 2. Generate profiled cost proto file by calling 'nsys stats'
# 3. Perform perf run by using the user-supplied xla_flags
# 4. Compute the percentage of collectives that are overlapping with compute kernels using generated profile from step 3.
# 5. Log (overlap_percentage, current_threshold), increment threshold and continue from step 1.
# 6. Return the most optimal threshold that yields larger overlap percentage

# The above sequence of operations is memory-intensive, so make sure to set the number of training steps to a small number.

# A sample command:
# python sweep_script.py --input_command="sh /pax/example_pp.sh" --output_dir="/pax/test_script" --initial_threshold=20480 
# --max_threshold=40960 --increment=5000 --xla_flags="--xla_gpu_enable_async_collective_permute=true --xla_gpu_simplify_all_fp_conversions 
#                                    --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_async_collective_permute=true 
#                                    --xla_disable_hlo_passes=collective-schedule-linearizer --xla_gpu_enable_async_reduce_scatter=true 
#                                    --xla_gpu_enable_async_all_gather=true --xla_gpu_enable_async_all_to_all=true --xla_gpu_enable_latency_hiding_scheduler=true"

import csv
import re
import sys
import argparse
import os
import shutil
import subprocess
import pip
from sortedcontainers import SortedList

def install(package):
    pip.main(['install', package])

try:
    import pandas as pd
except ModuleNotFoundError:
    print("module 'pandas' is not installed, installing...")
    install("pandas")
    import pandas as pd
pd.set_option('display.max_rows', 10)

nsys_path = shutil.which("nsys") 

if nsys_path is None:
  print("no nsys found on the system")
  exit()
else:
  print(f"path to executable nsys: {nsys_path}")

NSYS_PATH = nsys_path
WORK_DIR = os. getcwd()
DEFAULT_PROFILE_PREFIX = "profiled_run_"
INITIAL_THRESHOLD_VALUE = 20 * 1024 # 10KB
MAX_THRESHOLD_VALUE = 32 * 1024 * 1024 # 32MB, default value in XLA
DEFAULT_INCREMENT_VALUE = 5 * 1024 # Default increment by 5KB

result_list = SortedList()

parser = argparse.ArgumentParser(description='Tool to sweep for optimal allreduce combiner threshold')
parser.add_argument("--input_command", type=str, help="Input command, no need to add nsys, the script will add nsys profile in front of the command")
parser.add_argument("--output_dir", type=str, default=WORK_DIR, help="Output directoy of all profiles and pbtxt files")
parser.add_argument("--profile_prefix", type=str, default=DEFAULT_PROFILE_PREFIX, help="Profile's name prefix, all generated profiles will have this prefix.")
parser.add_argument("--initial_threshold", type=int, default=INITIAL_THRESHOLD_VALUE, help="Starting value for the sweeping. Default 10KB")
parser.add_argument("--max_threshold", type=int, default=MAX_THRESHOLD_VALUE, help="Ending value for the sweeping. Default 32MB")
parser.add_argument("--increment", type=int, default=DEFAULT_INCREMENT_VALUE, help="The incremented amount for each iteration. Default 5KB")
parser.add_argument("--xla_flags", type=str, required=False, help="xla flags to add to the profile runs, if all reduce combiner threshold is supplied, it will be overriden.")
parser.add_argument("--post_process", help="post process pbtxt to get minimum cost value for each instruction", action="store_true")
parser.add_argument("--debug", action="store_true")

args = parser.parse_args()
profile_xla_flags_to_set = []
perf_xla_flags_to_set = args.xla_flags.split(' ')

other_xla_flags = []
threshold_flag_name = '--xla_gpu_all_reduce_combine_threshold_bytes='
profile_output_path = args.output_dir

if not os.path.exists(profile_output_path):
  os.makedirs(profile_output_path)
  print("Created output path: {}".format(profile_output_path))

async_collective_flags = ['--xla_gpu_enable_async_collective_permute',
                    '--xla_gpu_enable_async_reduce_scatter',
                    '--xla_gpu_enable_async_all_reduce',
                    '--xla_gpu_enable_async_all_to_all',
                    '--xla_gpu_enable_async_all_gather']

if args.xla_flags:
  print("Found xla flags from input: {}".format(args.xla_flags))
  flags = args.xla_flags.split(' ')
  for flag in flags:
    if flag and not threshold_flag_name in flag:
      perf_xla_flags_to_set.append(flag)
      if not any(substring in flag for substring in async_collective_flags):
        other_xla_flags.append(flag)

  profile_xla_flags_to_set = other_xla_flags
# Explicitly Disable all async ops
disable_async_flags = []
for async_flag in async_collective_flags:
  disable_async_flags.append(async_flag + '=false')
profile_xla_flags_to_set = profile_xla_flags_to_set + disable_async_flags

current_threshold = args.initial_threshold
for _ in range(args.initial_threshold, (args.max_threshold + args.increment), args.increment):
  # Prepare profiling env and flags
  profile_full_path = os.path.join(profile_output_path, args.profile_prefix + str(current_threshold) + "_profiled")
  profile_run_command = [nsys_path, "profile", "-o", f"{profile_full_path}", "--force-overwrite", "true"]
  profile_xla_flags_to_set.append(threshold_flag_name + str(current_threshold))
  profile_xla_flag_env = " ".join(profile_xla_flags_to_set)
  profile_run_command = profile_run_command + args.input_command.split()
  print(f"""
  ******Starting profiling command******
  {profile_run_command}.
  ******With XLA flags******:
  {profile_xla_flag_env}""")
  profile_run_env = os.environ.copy()
  profile_run_env["XLA_FLAGS"] = profile_xla_flag_env
  proc = subprocess.Popen(profile_run_command, stdout=sys.stdout, stderr=sys.stderr, env=profile_run_env)
  proc.wait(timeout=1800)

  # Perform nsys stats to get kernel times
  stats_command = [nsys_path, "stats", "--force-overwrite", "true", "--force-export", "true", "--report", "nvtxkernsum", f"{profile_full_path}.nsys-rep", "-o", f"{profile_full_path}"]

  print(f"""
  ******Starting stats command******
  {stats_command}.
  """)

  proc = subprocess.Popen(stats_command, stdout=sys.stdout, stderr=sys.stderr)
  proc.wait(10 * 60)

  thunk_re = re.compile("TSL:Thunk:#hlo_op=(.*)#")
  cost_dictionary = dict()
  profile_proto_path = profile_full_path+".pbtxt"

  with open(profile_proto_path, 'w', newline='') as protofile:
    with open(profile_full_path+"_nvtxkernsum.csv", newline='') as csvfile:
      reader = csv.DictReader(csvfile)
      for row in reader:
        name = row['NVTX Range']

        time_ns = float(row['Avg (ns)'])
        m = thunk_re.match(name)
        if m is not None:
          if args.post_process:
            cost_dictionary.setdefault(m.group(1), []).append((time_ns))
          else:
            protofile.write(f'costs {{ name: "{m.group(1)}" cost_us: {time_ns / 1000.0} }}\n')
    if args.post_process:
      for name, cost in cost_dictionary.items():
        protofile.write(f'costs {{ name: "{name}" cost_us: {min(cost)} }}\n')

  # Prepare perf run xla flags
  perf_xla_flags_to_set.insert(0, "--xla_gpu_pgle_profile_file_or_directory_path=" + profile_proto_path)
  perf_profile_full_path = os.path.join(profile_output_path, args.profile_prefix + str(current_threshold) + "_perf_run")
  perf_run_command = [nsys_path, "profile", "-o", f"{perf_profile_full_path}", "--force-overwrite", "true"]
  perf_xla_flags_to_set.append(threshold_flag_name + str(current_threshold))
  perf_xla_flag_env = " ".join(perf_xla_flags_to_set)
  perf_run_command = perf_run_command + args.input_command.split()
  print(f"""
  ******Starting performance run command******
  {perf_run_command}.
  ******With XLA flags******:
  {perf_xla_flags_to_set}""")
  perf_run_env = os.environ.copy()
  perf_run_env["XLA_FLAGS"] = perf_xla_flag_env
  proc = subprocess.Popen(perf_run_command, stdout=sys.stdout, stderr=sys.stderr, env=perf_run_env)
  proc.wait(30 * 60)

  # Use profile of perf run to compute overlap percentage
  csv_path = perf_profile_full_path.split('.')[0]
  stats_trace_command = [nsys_path, "stats", "-o", f"{csv_path}", "--report", "cuda_gpu_trace", "--format", "csv", "--force-overwrite", "true", "--force-export", "true", f"{perf_profile_full_path}.nsys-rep"]
  print(f"""
  ******Starting stats GPU trace command******
  {stats_trace_command}.
  """)
  proc = subprocess.Popen(stats_trace_command, stdout=sys.stdout, stderr=sys.stderr)
  proc.wait(5 * 60)

  # Nsys appends (report type) to a given report name.
  # i.e. if "--report=cuda_gpu_trace", output report will have "_cuda_gpu_trace.csv" appended.
  csv_path = csv_path + "_cuda_gpu_trace.csv"

  input_df = pd.read_csv(csv_path)
  collective_df = input_df[input_df['Name'].str.contains('ncclKernel')]
  collective_df['end'] = collective_df['Start (ns)'] + collective_df['Duration (ns)']

  collective_intervals_list = list(zip(*map(collective_df.get, ['Start (ns)', 'end'])))

  # A naive way to get compute stream
  compute_stream = input_df[input_df['Name'].str.contains('fusion')]['Strm'].iloc[0]
  print("Compute stream: {}".format(compute_stream))

  compute_df = input_df[input_df['Strm'] == compute_stream]
  compute_df['end'] = compute_df['Start (ns)'] + compute_df['Duration (ns)']
  if args.debug:
    print("Compute df: \n {}".format(compute_df))

  compute_intervals_list = list(zip(*map(compute_df.get, ['Start (ns)', 'end'])))

  total_colletive_time = collective_df['Duration (ns)'].sum()
  if args.debug:
    print("Collective df: \n {}".format(collective_df))

  print("Total runtime of all collectives is: {}".format(total_colletive_time))

  i1 = 0
  i2 = 0
  overlapping = 0

  while i1 < len(compute_intervals_list) and i2 < len(collective_intervals_list):
    # start and end of the overlapping
    start = max(compute_intervals_list[i1][0], collective_intervals_list[i2][0]) 
    end = min(compute_intervals_list[i1][1], collective_intervals_list[i2][1])
    if args.debug:
      if end-start > 0:
        print("Found overlapping intervals between collective: {} and compute: {}".format(collective_intervals_list[i2], compute_intervals_list[i1]))

    overlapping += max(0, end - start)

    if compute_intervals_list[i1][1] < collective_intervals_list[i2][1]:
      i1 += 1
    else:
      i2 += 1

  overlap_percentage = float(overlapping / total_colletive_time)
  print("Total overlapping percentage is: {} % for threshold: {}".format(overlap_percentage*100, current_threshold))

  # Remove the temporary csv file.
  os.remove(csv_path)

  # Push the (percentage, threshold) tuple to the sorted list.
  result_list.add((overlap_percentage, current_threshold))
  # Remove old threshold flag
  profile_xla_flags_to_set.pop()
  perf_xla_flags_to_set.pop()
  current_threshold += args.increment

print("Sweeping run is done.")
best_percent, best_threshold = result_list.pop()
print("Best threshold was: {} which had {} overlapping percentage.".format(best_threshold, best_percent))

