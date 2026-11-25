import csv
import sys

with open(sys.argv[1], newline="") as source:
    rows = list(csv.reader(source))

header = rows[0]
metrics = [
    "gpu__time_duration.sum",
    "launch__grid_size",
    "launch__block_size",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_static",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
]

writer = csv.writer(sys.stdout)
writer.writerow(["kernel", "duration_us", "blocks", "threads", "registers",
                 "shared_kb", "occupancy", "load_conflicts", "store_conflicts"])
for row in rows[2:]:
    name = row[header.index("Kernel Name")].split("(")[0]
    writer.writerow([name] + [row[header.index(metric)] for metric in metrics])
