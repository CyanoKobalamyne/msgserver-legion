# Legion runtime directory must be specified for compilation.
ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

# Don't include debugging symbols by default.
DEBUG           ?= 0
# Set maximum number of dimensions.
MAX_DIM         ?= 3
# Set default logging level.
OUTPUT_LEVEL    ?= LEVEL_INFO

# Compute file names.
OUTFILE		?= messaging
GEN_SRC		?= $(OUTFILE).cc

# Use C++17 standard for better template type deduction.
CC_FLAGS	+= -std=c++17
# Increase maximum return size from 2048.
CC_FLAGS	+= -DLEGION_MAX_RETURN_SIZE=8192

# Include Legion's Makefile
include $(LG_RT_DIR)/runtime.mk
