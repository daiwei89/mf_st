MFST_ROOT := $(shell readlink $(dir $(lastword $(MAKEFILE_LIST))) -f)
SRC_DIR = $(MFST_ROOT)/src

MFST_THIRD_PARTY = $(MFST_ROOT)/third_party
MFST_THIRD_PARTY_SRC = $(MFST_THIRD_PARTY)/src
MFST_THIRD_PARTY_INCLUDE = $(MFST_THIRD_PARTY)/include
MFST_THIRD_PARTY_LIB = $(MFST_THIRD_PARTY)/lib

MFST_CXX = g++
MFST_CXXFLAGS = -g -O3 \
           -std=c++0x \
           -Wall \
					 -Wno-sign-compare \
           -fno-builtin-malloc \
           -fno-builtin-calloc \
           -fno-builtin-realloc \
           -fno-builtin-free \
           -fno-omit-frame-pointer
MFST_INCFLAGS = -I$(MFST_THIRD_PARTY_INCLUDE)
MFST_LDFLAGS = -Wl,-rpath,$(MFST_THIRD_PARTY_LIB) \
          -L$(MFST_THIRD_PARTY_LIB) \
          -pthread -lrt -lnsl -luuid \
          -lglog \
          -lgflags \
          -ltcmalloc

# defined in defns.mk
THIRD_PARTY = $(MFST_THIRD_PARTY)
THIRD_PARTY_SRC = $(MFST_THIRD_PARTY_SRC)
THIRD_PARTY_LIB = $(MFST_THIRD_PARTY_LIB)
THIRD_PARTY_INCLUDE = $(MFST_THIRD_PARTY_INCLUDE)

NEED_MKDIR = $(THIRD_PARTY_SRC) \
             $(THIRD_PARTY_LIB) \
             $(THIRD_PARTY_INCLUDE)

MFST_SRC = $(wildcard $(SRC_DIR)/*.cpp)
MFST_HDR = $(wildcard $(SRC_DIR)/*.hpp)
MFST_BIN = $(MFST_ROOT)/bin
MFST = $(MFST_BIN)/mf_main

mf: $(MFST)

$(MFST): $(MFST_SRC)
	mkdir -p $(MFST_BIN)
	$(MFST_CXX) $(MFST_CXXFLAGS) $(MFST_INCFLAGS) $^ \
	$(MFST_LDFLAGS) -o $@

$(MFST_OBJ): %.o: %.cpp $(MFST_HDR)
	$(MFST_CXX) $(MSFT_CXXFLAGS) -I$(SRC_DIR) $(INCFLAGS) -c $< -o $@


all: path \
     third_party_core

clean:
	rm -rf $(MFST_BIN)

.PHONY: all path mf clean

include $(THIRD_PARTY)/third_party.mk
