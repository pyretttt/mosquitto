#ifndef EVENT_LOOP_H
#define EVENT_LOOP_H

#include <stdint.h>

#include "consts.h"

enum class State: uint32_t { 
    Req = 0,
    Res = 1,
    End = 2,
};

struct Conn {
    int fd = -1;
    State state = State::Req;
    // buffer for reading
    size_t rbuf_size = 0;
    uint8_t rbuf[4 + k_max_msg];
    
    // buffer for writing
    size_t wbuf_size = 0;
    size_t wbuf_sent = 0;
    uint8_t wbuf[4 + k_max_msg];
};

#endif