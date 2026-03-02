
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Returns pointer to a null-terminated UTF-8 string owned by Rust.
// Valid for the lifetime of the program; do not free.
const char *rust_hello(void);

// Simple test function returning a constant number.
int32_t get_number(void);

#ifdef __cplusplus
} // extern "C"
#endif
