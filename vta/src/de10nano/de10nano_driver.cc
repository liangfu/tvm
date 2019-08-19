/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 * \file de10-nano_driver.cc
 * \brief VTA driver for DE10_Nano board.
 */

#include "de10nano_driver.h"

#include <vta/driver.h>
#include <dmlc/logging.h>
#include <thread>
#include "cma_api.h"

void* VTAMemAlloc(size_t size, int cached) {
  static int _ = cma_init(); (void)_;
  if (cached) {
    return cma_alloc_cached(size);
  } else {
    return cma_alloc_noncached(size);
  }
}

void VTAMemFree(void* buf) {
  cma_free(buf);
}

vta_phy_addr_t VTAMemGetPhyAddr(void* buf) {
  return cma_get_phy_addr(buf) + 0x80000000;
}

void VTAFlushCache(vta_phy_addr_t buf, int size) {
  CHECK(false) << "VTAFlushCache not implemented for de10nano";
}

void VTAInvalidateCache(vta_phy_addr_t buf, int size) {
  CHECK(false) << "VTAInvalidateCache not implemented for de10nano";
}

void *VTAMapRegister(uint32_t addr, size_t length) {
  // Align the base address with the pages
  uint32_t virt_base = addr & ~(getpagesize() - 1);
  // Calculate base address offset w.r.t the base address
  uint32_t virt_offset = addr - virt_base;
  // Open file and mmap
  uint32_t mmap_file = open(VTA_DE10NANO_DEV_MEM_PATH, O_RDWR|O_SYNC);
  return mmap(NULL,
              (length+virt_offset),
              PROT_READ|PROT_WRITE,
              MAP_SHARED,
              mmap_file,
              virt_base);
}

void VTAUnmapRegister(void *vta, size_t length) {
  // Unmap memory
  int status = munmap(vta, length);
  assert(status == 0);
}

void VTAWriteMappedReg(void* base_addr, uint32_t offset, uint32_t val) {
  *((volatile uint32_t *) (reinterpret_cast<char *>(base_addr) + offset)) = val;
}

uint32_t VTAReadMappedReg(void* base_addr, uint32_t offset) {
  return *((volatile uint32_t *) (reinterpret_cast<char *>(base_addr) + offset));
}

class VTADevice {
 public:
  VTADevice() {
    // VTA stage handles
    vta_host_handle_ = VTAMapRegister(VTA_HOST_ADDR, VTA_RANGE);
  }

  ~VTADevice() {
    // Close VTA stage handle
    VTAUnmapRegister(vta_host_handle_, VTA_RANGE);
  }

  int Run(vta_phy_addr_t insn_phy_addr,
          uint32_t insn_count,
          uint32_t wait_cycles) {
    VTAWriteMappedReg(vta_host_handle_, 0x04, 0);
    VTAWriteMappedReg(vta_host_handle_, 0x08, insn_count);
    VTAWriteMappedReg(vta_host_handle_, 0x0c, insn_phy_addr);

    // VTA start
    VTAWriteMappedReg(vta_host_handle_, 0x0, VTA_START);

    // Loop until the VTA is done
    unsigned t, flag = 0;
    for (t = 0; t < wait_cycles; ++t) {
      flag = VTAReadMappedReg(vta_host_handle_, 0x00);
      flag &= 0x2;
      if (flag == 0x2) break;
      std::this_thread::yield();
    }
    // Report error if timeout
    return t < wait_cycles ? 0 : 1;
  }

 private:
  // VTA handles (register maps)
  void* vta_host_handle_{nullptr};
};

VTADeviceHandle VTADeviceAlloc() {
  return new VTADevice();
}

void VTADeviceFree(VTADeviceHandle handle) {
  delete static_cast<VTADevice*>(handle);
}

int VTADeviceRun(VTADeviceHandle handle,
                 vta_phy_addr_t insn_phy_addr,
                 uint32_t insn_count,
                 uint32_t wait_cycles) {
  return static_cast<VTADevice*>(handle)->Run(
      insn_phy_addr, insn_count, wait_cycles);
}

void VTAProgram(const char* bitstream) {
  CHECK(false) << "VTAProgram not implemented for de10nano";
}
