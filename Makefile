# Set Legion runtime directory.
LG_RT_DIR=./legion/runtime

# Set flags for Legion.
DEBUG           ?= 1			# Include debugging symbols
MAX_DIM         ?= 3			# Maximum number of dimensions
OUTPUT_LEVEL    ?= LEVEL_INFO		# Compile time logging level

# Compute file names.
OUTFILE		?= messaging
GEN_SRC		?= $(OUTFILE).cc

# Set compilation flags
CC_FLAGS	?=	-std=c++17

# Include Legion's Makefile
include $(LG_RT_DIR)/runtime.mk
