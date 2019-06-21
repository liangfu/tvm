# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

DEBUG = 1

CC ?= g++
ifeq (${DEBUG},1)
  CFLAGS = -Wall -O0 -g -std=c++11
else
  CFLAGS = -Wall -O2 -std=c++11
endif
LDFLAGS =
EXECUTABLE = metal_test
INCLUDE_DIR = ../../../include
TESTLIB_DIR = ../common
LIBS = -lstdc++
SOURCES = test_lib.cc
OBJECTS = test_lib.o metal_test.o

VTA_CONFIG = python ../../../config/vta_config.py
VTA_TARGET := $(shell ${VTA_CONFIG} --target)

ifeq (${VTA_TARGET},pynq)
  CFLAGS += -I/usr/include
  LDFLAGS += -L/usr/lib -L/opt/python3.6/lib/python3.6/site-packages/pynq/lib/
  LIBS += -l:libsds_lib.so -l:libdma.so
  DRIVER_DIR = ../../../src/pynq
  SOURCES += pynq_driver.cc
  OBJECTS += pynq_driver.o
endif
ifeq (${VTA_TARGET},de10-nano)
  DRIVER_DIR = ../../../src/de10-nano
  SOURCES += de10-nano_driver.cc cma_api.cc
  OBJECTS += de10-nano_driver.o cma_api.o
endif

# Include VTA config
VPATH = $(DRIVER_DIR):$(TESTLIB_DIR)
CFLAGS += `${VTA_CONFIG} --cflags`
LDFLAGS += `${VTA_CONFIG} --ldflags`

# Define flags
CFLAGS += -I$(INCLUDE_DIR) -DNO_SIM -DVTA_DEBUG=0

# All Target
all: $(EXECUTABLE)

%.o: %.cc $(SOURCES)
	$(CC) -c -o $@ $< $(CFLAGS)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@ $(LIBS)

clean:
	rm -rf *.o $(EXECUTABLE)
