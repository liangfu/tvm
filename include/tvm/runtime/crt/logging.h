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

/*!
 * \file loggin.h
 * \brief A replacement of the dmlc logging system that avoids
 *  the usage of GLOG and C++ headers
 */

#ifndef TVM_RUNTIME_CRT_LOGGING_H_
#define TVM_RUNTIME_CRT_LOGGING_H_

#define CHECK(x)                                                        \
  do {                                                                  \
    if (!(x)) {                                                         \
      fprintf(stderr, "Check failed: %s\n", #x);                        \
      exit(-1);                                                         \
    }                                                                   \
  }while(0)

#define CHECK_BINARY_OP(op, x, y, fmt, ...)                             \
  do {                                                                  \
    if (!(x op y)) {                                                    \
      fprintf(stderr, "Check failed: %s %s %s: " fmt "\n", #x, #op, #y, ##__VA_ARGS__); \
      exit(-1);                                                         \
    }                                                                   \
  }while(0)

#define CHECK_LT(x, y, fmt, ...) CHECK_BINARY_OP(<,  x, y, fmt, ##__VA_ARGS__)
#define CHECK_GT(x, y, fmt, ...) CHECK_BINARY_OP(>,  x, y, fmt, ##__VA_ARGS__)
#define CHECK_LE(x, y, fmt, ...) CHECK_BINARY_OP(<=, x, y, fmt, ##__VA_ARGS__)
#define CHECK_GE(x, y, fmt, ...) CHECK_BINARY_OP(>=, x, y, fmt, ##__VA_ARGS__)
#define CHECK_EQ(x, y, fmt, ...) CHECK_BINARY_OP(==, x, y, fmt, ##__VA_ARGS__)
#define CHECK_NE(x, y, fmt, ...) CHECK_BINARY_OP(!=, x, y, fmt, ##__VA_ARGS__)

#endif  // TVM_RUNTIME_CRT_LOGGING_H_
