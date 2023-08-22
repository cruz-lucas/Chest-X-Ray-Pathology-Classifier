.PHONY: clean data lint format requirements sync_data_down sync_data_up

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = Chest X-Ray Pathology Classifier
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python
IMAGE_NAME = x-ray
CONTAINER_NAME = x-ray
REGISTRY = us-central1-docker.pkg.dev/labshurb/lucas-cruz-final-project
TAG = latest

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt


## Build docker image
docker_build:
	docker build -t $(IMAGE_NAME):$(TAG) -f Dockerfile .


## Run docker interactively
docker_run:
	docker run -it --gpus all --name $(CONTAINER_NAME) --rm $(IMAGE_NAME)


## Push image to cloud
docker_push:
	docker tag $(IMAGE_NAME):$(TAG) $(REGISTRY)/xray:latest
	docker push $(REGISTRY)/xray:latest


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using flake8 and black (use `make format` to do formatting)
lint:
	flake8 src
	black --check --config pyproject.toml src


## Format source code with black
format:
	black --config pyproject.toml src


## Download Data from storage system
sync_data_down:
	gsutil rsync gs://bucket-name/data/ data/
	

## Upload Data to storage system
sync_data_up:
	gsutil rsync data/ gs://bucket-name/data/


## Set up python interpreter environment
create_environment:
	$(PYTHON_INTERPRETER) -m venv env


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## train local
train:
	$(PYTHON_INTERPRETER) src/train.py 	
# --input_filepath "gcs://chexpert_database_stanford/"


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
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
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
