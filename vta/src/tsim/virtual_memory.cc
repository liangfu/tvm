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
 */

#include "virtual_memory.h"

#include <dmlc/logging.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <list>
#include <utility>
#include <iterator>
#include <unordered_map>

/*! page size of virtual address */
#ifndef VTA_TSIM_VM_PAGE_SIZE
#define VTA_TSIM_VM_PAGE_SIZE (4096)
#endif  // VTA_TSIM_VM_PAGE_SIZE

/*! starting point of the virtual address system */
#ifndef VTA_TSIM_VM_ADDR_BEGIN
#define VTA_TSIM_VM_ADDR_BEGIN (0x40000000)
#endif  // VTA_TSIM_VM_ADDR_BEGIN

namespace vta {
namespace tsim {

static const uint64_t kPageSize = VTA_TSIM_VM_PAGE_SIZE;
static const uint64_t kVirtualMemoryOffset = VTA_TSIM_VM_ADDR_BEGIN;

class VirtualMemoryManager {
  /*! \brief page table */
  std::list<std::pair<uint64_t, uint64_t> > _page_table;
  /*! \brief translation lookaside buffer */
  std::unordered_map<uint64_t, size_t> _tlb;

 public:
  /*! \brief allocate virtual memory for given size */
  void * Allocate(uint64_t sz) {
    uint64_t size = ((sz + kPageSize - 1) / kPageSize) * kPageSize;
    void * ptr = malloc(size);
    size_t laddr = reinterpret_cast<size_t>(ptr);
    // search for available virtual memory space
    uint64_t vaddr = kVirtualMemoryOffset;
    auto it = _page_table.begin();
    if (((*it).first - kVirtualMemoryOffset) < size) {
      it++;
      for (; it != _page_table.end(); it++) {
        if (((*it).first - (*std::prev(it)).second) > size) {
          vaddr = (*std::prev(it)).second;
        }
      }
      if (it == _page_table.end()) {
        vaddr = (*std::prev(it)).second;
      }
    }
    _page_table.push_back(std::make_pair(vaddr, vaddr + size));
    _tlb[vaddr] = laddr;
    // save tlb to file in order to be accessed externally.
    FILE * fout = fopen(VTA_VMEM_PAGEFILE, "wb");
    CHECK(fout);
    if (fout) {
      uint32_t tlb_size = sizeof(uint64_t)*3*_tlb.size();
      uint32_t tlb_cnt = 0;
      uint64_t * tlb = new uint64_t[3*_tlb.size()];
      for (auto iter = _tlb.begin(); iter != _tlb.end(); iter++, tlb_cnt++) {
        tlb[tlb_cnt * 3] = (*iter).first;
        uint64_t vend = 0;
        for (auto iter_in = _page_table.begin(); iter_in != _page_table.end(); iter_in++) {
          if ((*iter_in).first == (*iter).first) { vend = (*iter_in).second; break; }
        }
        tlb[tlb_cnt * 3 + 1] = vend;
        tlb[tlb_cnt * 3 + 2] = (*iter).second;
      }
      fwrite(tlb, tlb_size, 1, fout);
      fflush(fout);
      fclose(fout);
    }
    // return virtual address
    return reinterpret_cast<void*>(vaddr);
  }

  /*! \brief release virtual memory for pointer */
  void Release(void * ptr) {
    uint64_t src = reinterpret_cast<uint64_t>(ptr);
    auto it = _page_table.begin();
    for (; it != _page_table.end(); it++) {
      if (((*it).first <= src) && ((*it).second > src)) { break; }
    }
    CHECK(it != _page_table.end());
    uint64_t * laddr = reinterpret_cast<uint64_t*>(_tlb[(*it).first]);
    delete [] laddr;
    _page_table.erase(it);
    _tlb.erase((*it).first);
  }

  /*! \brief copy virtual memory from host */
  void MemCopyFromHost(void * dstptr, const void * src, uint64_t size) {
    // get logical address from virtual address
    size_t dst = reinterpret_cast<size_t>(dstptr);
    auto it = _page_table.begin();
    for (; it != _page_table.end(); it++) {
      if (((*it).first <= dst) && ((*it).second > dst)) { break; }
    }
    CHECK(it != _page_table.end());
    size_t offset = dst - (*it).first;
    char * laddr = reinterpret_cast<char*>(_tlb[(*it).first]);
    // copy content from src to logic address
    memcpy(laddr + offset, src, size);
  }

  /*! \brief copy virtual memory to host */
  void MemCopyToHost(void * dst, const void * srcptr, uint64_t size) {
    // get logical address from virtual address
    size_t src = reinterpret_cast<size_t>(srcptr);
    auto it = _page_table.begin();
    for (; it != _page_table.end(); it++) {
      if (((*it).first <= src) && ((*it).second > src)) { break; }
    }
    CHECK(it != _page_table.end());
    size_t offset = src - (*it).first;
    char * laddr = reinterpret_cast<char*>(_tlb[(*it).first]);
    // copy content from logic address to dst
    memcpy(dst, laddr + offset, size);
  }

  /*! \brief get logical address from virtual memory */
  void * GetLogicalAddr(uint64_t src) {
    if (src == 0) { return 0; }
    auto it = _page_table.begin();
    for (; it != _page_table.end(); it++) {
      if (((*it).first <= src) && ((*it).second > src)) { break; }
    }
    CHECK(it != _page_table.end());
    return reinterpret_cast<void*>(_tlb[(*it).first]);
  }

  /*! \brief get global handler of the instance */
  static VirtualMemoryManager* Global() {
    static VirtualMemoryManager inst;
    return &inst;
  }
};

}  // namespace tsim
}  // namespace vta


void * vmalloc(uint64_t size) {
  return vta::tsim::VirtualMemoryManager::Global()->Allocate(size);
}

void vfree(void * ptr) {
  vta::tsim::VirtualMemoryManager::Global()->Release(ptr);
}

void vmemcpy(void * dst, const void * src, uint64_t size, VMemCopyType dir) {
  auto * mgr = vta::tsim::VirtualMemoryManager::Global();
  CHECK(mgr != 0);
  if (kVirtualMemCopyFromHost == dir) {
    mgr->MemCopyFromHost(dst, src, size);
  } else {
    mgr->MemCopyToHost(dst, src, size);
  }
}

void * vmem_get_log_addr(uint64_t vaddr) {
  return vta::tsim::VirtualMemoryManager::Global()->GetLogicalAddr(vaddr);
}
