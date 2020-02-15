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
 * \file load_json.h
 * \brief Lightweight JSON Reader that read save into C++ data structs.
 */
#ifndef TVM_RUNTIME_CRT_LOAD_JSON_H_
#define TVM_RUNTIME_CRT_LOAD_JSON_H_

#include <stdio.h>
#include <ctype.h>

enum {
  JSON_READ_TYPE_U8 = 1,
  JSON_READ_TYPE_S8 = 2,
  JSON_READ_TYPE_U16 = 3,
  JSON_READ_TYPE_S16 = 4,
  JSON_READ_TYPE_U32 = 5,
  JSON_READ_TYPE_S32 = 6,
  JSON_READ_TYPE_F32 = 7,
  JSON_READ_TYPE_F64 = 8,
  JSON_READ_TYPE_GRAPH_RUNTIME_NODE = 9,
  JSON_READ_TYPE_GRAPH_RUNTIME_NODE_ENTRY = 10,
  JSON_READ_TYPE_GRAPH_RUNTIME_GRAPH_ATTR = 11
};

typedef struct seq_t {
  uint32_t * data;
  uint64_t allocated;
  uint32_t size;
  void (*push_back)(struct seq_t * seq, uint32_t src);
  uint32_t * (*back)(struct seq_t * seq);
  void (*pop_back)(struct seq_t * seq);
} Seq;

static inline void SeqPush(Seq * seq, uint32_t src) {
  if (seq->size >= seq->allocated) {
    printf("seq too large.\n");
  }
  seq->data[seq->size] = src;
  seq->size += 1;
}

static inline uint32_t * SeqBack(Seq * seq) {
  if (seq->size >= seq->allocated) {
    printf("seq too large.\n");
  }
  return seq->data + (seq->size-1);
}

static inline void SeqPop(Seq * seq) {
  if (seq->size >= seq->allocated) {
    printf("seq size is too large.\n");
  }
  if (seq->size == 0) {
    printf("seq size is too small.\n");
  }
  seq->size -= 1;
}

static inline Seq * SeqCreate(uint64_t len) {
  Seq * seq = (Seq*)malloc(sizeof(Seq));  // NOLINT(*)
  memset(seq, 0, sizeof(Seq));
  seq->allocated = len;
  seq->data = (uint32_t*)malloc(sizeof(uint32_t)*len);  // NOLINT(*)
  seq->push_back = SeqPush;
  seq->back = SeqBack;
  seq->pop_back = SeqPop;
  return seq;
}

static inline void SeqRelease(Seq ** seq) {
  free((*seq)->data);
  free(*seq);
}

/*!
 * \brief Lightweight JSON Reader to read any STL compositions and structs.
 *  The user need to know the schema of the
 */
typedef struct json_reader_t {
  /*! \brief internal reader string */
  /* char is_[TVM_CRT_MAX_JSON_LENGTH]; */
  char * is_;
  char * isptr;
  /*! \brief "\\r" counter */
  size_t line_count_r_;
  /*! \brief "\\n" counter */
  size_t line_count_n_;
  /*!
   * \brief record how many element processed in
   *  current array/object scope.
   */
  Seq * scope_counter_;

  char (*NextChar)(struct json_reader_t * reader);
  char (*NextNonSpace)(struct json_reader_t * reader);
  char (*PeekNextChar)(struct json_reader_t * reader);
  char (*PeekNextNonSpace)(struct json_reader_t * reader);
  int (*ReadUnsignedInteger)(struct json_reader_t * reader, unsigned int * out_value);
  int (*ReadInteger)(struct json_reader_t * reader, int64_t * out_value);
  int (*ReadString)(struct json_reader_t * reader, char * out_value);
  void (*BeginArray)(struct json_reader_t * reader);
  void (*BeginObject)(struct json_reader_t * reader);
  uint8_t (*NextObjectItem)(struct json_reader_t * reader, char * out_key);
  uint8_t (*NextArrayItem)(struct json_reader_t * reader);
} JSONReader;

typedef void (*ReadFunction)(JSONReader *reader, void *addr);

/*! \brief internal data entry */
struct JSONObjectReadHelperEntry {
  /*! \brief the reader function */
  ReadFunction func;
  /*! \brief the address to read */
  void *addr;
  /*! \brief whether it is optional */
  uint8_t optional;
};

/*!
 * \brief Helper class to read JSON into a class or struct object.
 * \code
 *  struct Param {
 *    string name;
 *    int value;
 *    // define load function from JSON
 *    inline void Load(dmlc::JSONReader *reader) {
 *      dmlc::JSONStructReadHelper helper;
 *      helper.DeclareField("name", &name);
 *      helper.DeclareField("value", &value);
 *      helper.ReadAllFields(reader);
 *    }
 *  };
 * \endcode
 */
struct JSONObjectReadHelper {
  /*!
   * \brief Read in all the declared fields.
   * \param reader the JSONReader to read the json.
   */
  void (*ReadAllFields)(JSONReader *reader);
  /*!
   * \brief The internal reader function.
   * \param reader The reader to read.
   * \param addr The memory address to read.
   */
  void (*ReaderFunction)(JSONReader *reader, void *addr);
};

#define DMLC_JSON_ENABLE_ANY_VAR_DEF(KeyName)                  \
  static DMLC_ATTRIBUTE_UNUSED ::dmlc::json::AnyJSONManager&   \
  __make_AnyJSONType ## _ ## KeyName ## __

/*!
 * \def DMLC_JSON_ENABLE_ANY
 * \brief Macro to enable save/load JSON of dmlc:: whose actual type is Type.
 * Any type will be saved as json array [KeyName, content]
 *
 * \param Type The type to be registered.
 * \param KeyName The Type key assigned to the type, must be same during load.
 */
#define DMLC_JSON_ENABLE_ANY(Type, KeyName)                             \
  DMLC_STR_CONCAT(DMLC_JSON_ENABLE_ANY_VAR_DEF(KeyName), __COUNTER__) = \
    ::dmlc::json::AnyJSONManager::Global()->EnableType<Type>(#KeyName) \

// implementations of JSONReader

/*!
 * \brief Takes the next char from the input source.
 * \return the next character.
 */
static inline char JSONReader_NextChar(JSONReader * reader) {
  char ch = reader->isptr[0];
  reader->isptr += 1;
  return ch;
}

/*!
 * \brief Returns the next char from the input source.
 * \return the next character.
 */
static inline char JSONReader_PeekNextChar(JSONReader * reader) {
  return reader->isptr[0];
}

/*!
 * \brief Read next nonspace character.
 * \return the next nonspace character.
 */
static inline char JSONReader_NextNonSpace(JSONReader * reader) {
  int ch;
  do {
    ch = reader->NextChar(reader);
    if (ch == '\n') { ++(reader->line_count_n_); }
    if (ch == '\r') { ++(reader->line_count_r_); }
  } while (isspace(ch));
  return ch;
}

/*!
 * \brief Read just before next nonspace but not read that.
 * \return the next nonspace character.
 */
static inline char JSONReader_PeekNextNonSpace(JSONReader * reader) {
  int ch;
  while (1) {
    ch = reader->PeekNextChar(reader);
    if (ch == '\n') { ++(reader->line_count_n_); }
    if (ch == '\r') { ++(reader->line_count_r_); }
    if (!isspace(ch)) break;
    reader->NextChar(reader);
  }
  return ch;
}

/*!
 * \brief Parse next JSON string.
 * \param out_str the output string.
 * \throw dmlc::Error when next token is not string
 */
static inline int JSONReader_ReadString(JSONReader * reader, char * out_str) {
  int status = 0;
  char ch = reader->NextNonSpace(reader);
  char output[128];
  uint32_t output_counter = 0;
  memset(output, 0, 128);
  while (1) {
    ch = reader->NextChar(reader);
    if (ch == '\\') {
      char sch = reader->NextChar(reader);
      switch (sch) {
      case 'r': snprintf(output, sizeof(output), "%s\r", output); break;
      case 'n': snprintf(output, sizeof(output), "%s\n", output); break;
      case '\\': snprintf(output, sizeof(output), "%s\\", output); break;
      case 't': snprintf(output, sizeof(output), "%s\t", output); break;
      case '\"': snprintf(output, sizeof(output), "%s\"", output); break;
      default: fprintf(stderr, "unknown string escape %c\n", sch);
      }
    } else {
      if (ch == '\"') { break; }
      if (strlen(output) >= 127) {
        fprintf(stderr, "Error: detected buffer overflow.\n");
        status = -1;
        break;
      }
      strncat(output, &ch, 1);
      output_counter++;
      if (output_counter >= 127) {
        fprintf(stderr, "Error: string size greater than 128.\n");
        status = -1;
        break;
      }
    }
    if (ch == EOF || ch == '\r' || ch == '\n') {
      fprintf(stderr, "Error at line X, Expect \'\"\' but reach end of line\n");
    }
  }
  snprintf(out_str, sizeof(output), "%s", output);
  return status;
}

static inline int JSONReader_ReadUnsignedInteger(JSONReader * reader, unsigned int * out_value) {
  int status = 0;
  char* endptr;
  const char* icstr = reader->isptr;
  unsigned int number = strtol(icstr, &endptr, 10);
  reader->isptr += endptr - icstr;
  *out_value = number;
  return status;
}


static inline int JSONReader_ReadInteger(JSONReader * reader, int64_t * out_value) {
  int status = 0;
  char* endptr;
  const char* icstr = reader->isptr;
  int64_t number = strtol(icstr, &endptr, 10);
  reader->isptr += endptr - icstr;
  *out_value = number;
  return status;
}

/*!
 * \brief Begin parsing an object.
 * \code
 *  string key;
 *  // value can be any type that is json serializable.
 *  string value;
 *  reader->BeginObject();
 *  while (reader->NextObjectItem(&key)) {
 *    // do somthing to key value
 *    reader->Read(&value);
 *  }
 * \endcode
 */
static inline void JSONReader_BeginObject(JSONReader * reader) {
  int ch = reader->NextNonSpace(reader);
  if (!(ch == '{')) {
    fprintf(stderr, "Error at line X, Expect \'{\' but got \'%c\'\n", ch);
  }
  Seq * scope_counter_ = reader->scope_counter_;
  scope_counter_->push_back(scope_counter_, 0);
}

/*!
 * \brief Try to move to next object item.
 *  If this call is successful, user can proceed to call
 *  reader->Read to read in the value.
 * \param out_key the key to the next object.
 * \return true if the read is successful, false if we are at end of the object.
 */
static inline uint8_t JSONReader_NextObjectItem(JSONReader * reader, char * out_key) {
  uint8_t next = 1;
  Seq * scope_counter_ = reader->scope_counter_;
  if (scope_counter_->back(scope_counter_)[0] != 0) {
    int ch = reader->NextNonSpace(reader);
    if (ch == EOF) {
      next = 0;
    } else if (ch == '}') {
      next = 0;
    } else {
      if (ch != ',') {
        fprintf(stderr, "Error at line X, JSON object expect \'}\' or \',\' but got \'%c\'\n", ch);
      }
    }
  } else {
    int ch = reader->PeekNextNonSpace(reader);
    if (ch == '}') {
      reader->NextChar(reader);
      next = 0;
    }
  }
  if (!next) {
    scope_counter_->pop_back(scope_counter_);
    return 0;
  } else {
    scope_counter_->back(scope_counter_)[0] += 1;
    reader->ReadString(reader, out_key);
    int ch = reader->NextNonSpace(reader);
    if (ch != ':') {
      fprintf(stderr, "Error at line X, Expect \':\' but get \'%c\'\n", ch);
    }
    return 1;
  }
}

/*!
 * \brief Begin parsing an array.
 * \code
 *  // value can be any type that is json serializable.
 *  string value;
 *  reader->BeginArray();
 *  while (reader->NextArrayItem(&value)) {
 *    // do somthing to value
 *  }
 * \endcode
 */
static inline void JSONReader_BeginArray(JSONReader * reader) {
  int ch = reader->NextNonSpace(reader);
  if (ch != '[') {
    fprintf(stderr, "Error at line X, Expect \'[\' but get \'%c\'\n", ch);
  }
  Seq * scope_counter_ = reader->scope_counter_;
  scope_counter_->push_back(scope_counter_, 0);
}

/*!
 * \brief Try to read the next element in the array.
 *  If this call is successful, user can proceed to call
 *  reader->Read to read in the value.
 * \return true if the read is successful, false if we are at end of the array.
 */
static inline uint8_t JSONReader_NextArrayItem(JSONReader * reader) {
  uint8_t next = 1;
  Seq * scope_counter_ = reader->scope_counter_;
  if (scope_counter_->back(scope_counter_)[0] != 0) {
    int ch = reader->NextNonSpace(reader);
    if (ch == EOF) {
      next = 0;
    } else if (ch == ']') {
      next = 0;
    } else {
      if (ch != ',') {
        fprintf(stderr, "Error at line X, JSON object expect \']\' or \',\' but got \'%c\'\n", ch);
      }
    }
  } else {
    int ch = reader->PeekNextNonSpace(reader);
    if (ch == ']') {
      reader->NextChar(reader);
      next = 0;
    }
  }
  if (!next) {
    scope_counter_->pop_back(scope_counter_);
    return 0;
  } else {
    scope_counter_->back(scope_counter_)[0] += 1;
    return 1;
  }
}

/*!
 * \brief Constructor.
 * \param is the input source.
 */
static inline JSONReader JSONReader_Create(const char * is) {
  JSONReader reader;
  memset(&reader, 0, sizeof(JSONReader));
  reader.scope_counter_ = SeqCreate(200);
  reader.NextChar = JSONReader_NextChar;
  reader.PeekNextChar = JSONReader_PeekNextChar;
  reader.NextNonSpace = JSONReader_NextNonSpace;
  reader.PeekNextNonSpace = JSONReader_PeekNextNonSpace;
  reader.ReadString = JSONReader_ReadString;
  reader.ReadUnsignedInteger = JSONReader_ReadUnsignedInteger;
  reader.ReadInteger = JSONReader_ReadInteger;
  reader.BeginArray = JSONReader_BeginArray;
  reader.BeginObject = JSONReader_BeginObject;
  reader.NextArrayItem = JSONReader_NextArrayItem;
  reader.NextObjectItem = JSONReader_NextObjectItem;
  reader.is_ = (char*)malloc(strlen(is)+1);  // NOLINT(*)
  memset(reader.is_, 0, strlen(is)+1);
  snprintf(reader.is_, strlen(is)+1, "%s", is);
  reader.isptr = reader.is_;
  return reader;
}

static inline void JSONReader_Release(JSONReader * reader) {
  SeqRelease(&(reader->scope_counter_));
  free(reader->is_);
}

#endif  // TVM_RUNTIME_CRT_LOAD_JSON_H_
