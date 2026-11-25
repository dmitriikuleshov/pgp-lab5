NVCC := /usr/local/cuda-13.4/bin/nvcc
NVCC_FLAGS := -Werror cross-execution-space-call -lm -O2 -lineinfo

SRC_DIR := src/cuda
BUILD_DIR := build
REPORT_DIR := report

NCU := /opt/nvidia/nsight-compute/2026.3.1/ncu
PROFILE_SIZE := 10000000
PROFILE_INPUT := data/in/profile_$(PROFILE_SIZE).data

TARGETS := \
	$(BUILD_DIR)/lab5 \
	$(BUILD_DIR)/benchmark

.PHONY: all clean benchmark profile report

all: $(TARGETS)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/lab5: $(SRC_DIR)/lab5.cu $(SRC_DIR)/helper_cuda.h | $(BUILD_DIR)
	@echo "  NVCC      $@"
	@$(NVCC) $(NVCC_FLAGS) $< -o $@

$(BUILD_DIR)/benchmark: $(SRC_DIR)/benchmark.cu $(SRC_DIR)/helper_cuda.h | $(BUILD_DIR)
	@echo "  NVCC      $@"
	@$(NVCC) $(NVCC_FLAGS) $< -o $@

benchmark: $(BUILD_DIR)/benchmark
	@echo "  BENCHMARK $(REPORT_DIR)/benchmark_results.txt"
	@$(BUILD_DIR)/benchmark > $(REPORT_DIR)/benchmark_results.txt

$(PROFILE_INPUT): src/generate_array.py
	@echo "  GENERATE  $@"
	@mkdir -p data/in
	@python3 $< $(PROFILE_SIZE) > $@

profile: $(BUILD_DIR)/lab5 $(PROFILE_INPUT) src/profile_summary.py
	@echo "  NCU       $(REPORT_DIR)/profile/lab5.ncu-repz"
	@mkdir -p $(REPORT_DIR)/profile
	@$(NCU) --set full --launch-count 3 --force-overwrite \
		--export $(REPORT_DIR)/profile/lab5 \
		$(BUILD_DIR)/lab5 < $(PROFILE_INPUT) > /dev/null
	@$(NCU) --import $(REPORT_DIR)/profile/lab5.ncu-repz \
		--page raw --csv > $(REPORT_DIR)/profile/metrics.csv
	@$(NCU) --import $(REPORT_DIR)/profile/lab5.ncu-repz \
		--page details --print-details all > $(REPORT_DIR)/profile/lab5.txt
	@python3 src/profile_summary.py $(REPORT_DIR)/profile/metrics.csv \
		> $(REPORT_DIR)/profile_results.csv

report: benchmark profile
	@echo "  TYPST     $(REPORT_DIR)/report.pdf"
	@typst compile --font-path /mnt/c/Windows/Fonts \
		$(REPORT_DIR)/report.typ $(REPORT_DIR)/report.pdf

clean:
	@echo "  CLEAN"
	@rm -f $(TARGETS)
	@rm -f $(REPORT_DIR)/benchmark_results.txt
	@rm -f $(REPORT_DIR)/report.pdf
	@rm -f $(REPORT_DIR)/profile_results.csv
