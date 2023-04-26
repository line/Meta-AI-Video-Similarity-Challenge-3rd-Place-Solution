.PHONY: build pull pack-submission test-submission update-submodules data-subset force-pack-submission copy-full-results eval-train-subset
# ================================================================================================
# Settings
# ================================================================================================

ifeq (, $(shell which nvidia-smi))
CPU_OR_GPU ?= cpu
else
CPU_OR_GPU ?= gpu
endif

ifeq (${CPU_OR_GPU}, gpu)
GPU_ARGS = --gpus all
endif

SKIP_GPU ?= false
ifeq (${SKIP_GPU}, true)
GPU_ARGS =
endif

TAG = ${CPU_OR_GPU}-latest
LOCAL_TAG = ${CPU_OR_GPU}-local

REPO = meta-vsc-matching-runtime
REGISTRY_IMAGE = metavsc.azurecr.io/${REPO}:${TAG}
LOCAL_IMAGE = ${REPO}:${LOCAL_TAG}
CONTAINER_NAME = competition-meta-vsc

DATASET?=test
SUBSET_PROPORTION?=0.01
FULL_RESULTS_DIR?=./full_results
COMPETITION_DATA?=./data

# if not TTY (for example GithubActions CI) no interactive tty commands for docker
ifneq (true, ${GITHUB_ACTIONS_NO_TTY})
TTY_ARGS = -it
endif

# option to block or allow internet access from the submission Docker container
ifeq (true, ${BLOCK_INTERNET})
NETWORK_ARGS = --network none
endif

# To run a submission, use local version if that exists; otherwise, use official version
# setting SUBMISSION_IMAGE as an environment variable will override the image
SUBMISSION_IMAGE ?= $(shell docker images -q ${LOCAL_IMAGE})
ifeq (,${SUBMISSION_IMAGE})
SUBMISSION_IMAGE := $(shell docker images -q ${REGISTRY_IMAGE})
endif

# Give write access to the submission folder to everyone so Docker user can write when mounted
_submission_write_perms:
	chmod -R 0777 submission/

# ================================================================================================
# Commands for building the container if you are changing the requirements
# ================================================================================================

## Builds the container locally
build:
	docker build \
		--build-arg CPU_OR_GPU=${CPU_OR_GPU} \
		-t ${LOCAL_IMAGE} \
		-f runtime/Dockerfile .

## Fetch or update all submodules (vsc2022 and VCSL)
update-submodules:
	git pull && \
	git submodule update --init --recursive

## Ensures that your locally built container can import all the Python packages successfully when it runs
test-container: build _submission_write_perms
	docker run ${GPU_ARGS} \
		${TTY_ARGS} \
		--mount type=bind,source="$(shell pwd)"/runtime/tests,target=/tests,readonly \
		${LOCAL_IMAGE} \
		/bin/bash -c "conda run --no-capture-output -n condaenv pytest tests/test_packages.py"

## Start your locally built container and open a bash shell within the running container; same as submission setup except has network access
interact-container: build _submission_write_perms
	docker run ${GPU_ARGS}\
		--mount type=bind,source="$(shell pwd)"/data/${DATASET},target=/data,readonly \
		--mount type=bind,source="$(shell pwd)"/submission,target=/code_execution/submission \
		--mount type=bind,source="$(shell pwd)"/submission_src,target=/code_execution/submission_src \
		--mount type=bind,source=${COMPETITION_DATA},target=${COMPETITION_DATA} \
		--shm-size 8g \
		-it \
		${LOCAL_IMAGE} \
		/bin/bash

## Pulls the official container from Azure Container Registry
pull:
	docker pull ${REGISTRY_IMAGE}

## Creates a submission/submission.zip file from the source code in submission_quickstart
pack-quickstart:
# Don't overwrite so no work is lost accidentally
ifneq (,$(wildcard ./submission/submission.zip))
	$(error You already have a submission/submission.zip file. Rename or remove that file (e.g., rm submission/submission.zip).)
endif
	python scripts/generate_valid_random_matches.py && \
	cd submission_quickstart; zip -r ../submission/submission.zip main.py full_matches.csv

## Creates a submission/submission.zip file from the source code in submission_src
pack-submission:
# Don't overwrite so no work is lost accidentally
ifneq (,$(wildcard ./submission/submission.zip))
	$(error You already have a submission/submission.zip file. Rename or remove that file (e.g., rm submission/submission.zip).)
endif
	cd submission_src; zip -r ../submission/submission.zip ./*


## Runs container using code from `submission/submission.zip` and data from `data/test`. Can use `data/train` with DATASET=train.
test-submission: _submission_write_perms
# if submission file does not exist
ifeq (,$(wildcard ./submission/submission.zip))
	$(error To test your submission, you must first put a "submission.zip" file in the "submission" folder. \
	  If you want to use the benchmark, you can run `make pack-benchmark` first)
endif

# if container does not exist, error and tell user to pull or build
ifeq (${SUBMISSION_IMAGE},)
	$(error To test your submission, you must first run `make pull` (to get official container) or `make build` \
		(to build a local version if you have changes).)
endif
	docker run \
		${TTY_ARGS} \
		${GPU_ARGS} \
		${NETWORK_ARGS} \
		--network none \
		--mount type=bind,source="$(shell pwd)"/data/${DATASET},target=/data,readonly \
		--mount type=bind,source="$(shell pwd)"/submission,target=/code_execution/submission \
		--shm-size 8g \
		--cpuset-cpus 0-5 \
		--name ${CONTAINER_NAME} \
		--rm \
		${SUBMISSION_IMAGE}


## Adds video metadata and a subset of query videos to `data`. Defaults to test set. Can use train set with DATASET=train.
data-subset: _clean_subset_data
	python scripts/generate_data_subset.py --dataset ${DATASET} --subset_proportion ${SUBSET_PROPORTION}


_clean_subset_data:
	rm -f data/${DATASET}/query_subset.csv data/${DATASET}/subset_ground_truth.csv data/query/*.mp4


## Delete temporary Python cache and bytecode files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


copy-full-results:
	sudo cp ${FULL_RESULTS_DIR}/matches.csv submission_src/full_matches.csv
	sudo cp ${FULL_RESULTS_DIR}/processed_ref.npz submission_src/src/reference.npz
	sudo cp ${FULL_RESULTS_DIR}/noise.npz submission_src/src/noise.npz
	sudo cp ${FULL_RESULTS_DIR}/pca_matrix.bin submission_src/src/pca_matrix.bin


force-pack-submission:
	sudo rm -f submission/*
	cd submission_src; zip -r ../submission/submission.zip ./*


eval-train-subset: DATASET = train
eval-train-subset: force-pack-submission test-submission
	cd submission; tar -xvzf submission.tar.gz; cd ../
	python vsc2022/matching_eval.py \
		--predictions submission/subset_matches.csv \
		--ground_truth data/train/subset_ground_truth.csv


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo
	@echo "$$(tput bold)Settings based on your machine:$$(tput sgr0)"
	@echo SUBMISSION_IMAGE=${SUBMISSION_IMAGE}  "\t# ID of the image that will be used when running test-submission"
	@echo
	@echo "$$(tput bold)Available competition images:$$(tput sgr0)"
	@echo "$(shell docker images --format '{{.Repository}}:{{.Tag}} ({{.ID}}); ' ${REPO})"
	@echo
	@echo "$$(tput bold)Available commands:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
