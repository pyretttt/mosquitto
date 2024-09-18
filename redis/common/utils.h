#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <fcntl.h>

inline int32_t read_full(int fd, char *buff, size_t n) {
    while (n > 0) {
        ssize_t rv = read(fd, buff, n);
        if (rv <= 0) {
            return -1;
        }
        assert((size_t) rv <= n);
        n -= (size_t) rv;
        buff += rv;
    }
    return 0;
}


inline int32_t write_all(int fd, const char *buff, size_t n) {
    while (n > 0) {
        ssize_t rv = write(fd, buff, n);
        if (rv <= 0) {
            return -1;
        }
        assert((size_t) rv <= n);
        n -= (size_t) rv;
        buff += rv;
    }
    return 0;
}

inline void fd_to_nb(int fd) {
    errno = 0;
    int flags = fcntl(fd, F_GETFL, 0);
    if (errno) {
        std::cerr << "Failed to get fd flags" << std::endl;
        return;
    }

    flags |= O_NONBLOCK;

    errno = 0;
    fcntl(fd, F_SETFL, flags);
    if (errno) {
        std::cerr << "Failed to set fd flags" << std::endl;
    }
}