#pragma once

#include <iostream>
#include <variant>
#include <vector>

#include "GL/glew.h"

#include "Core.hpp"

namespace gl {
    using ID = unsigned int;

    // Required because by default all uniforms are initialized to 0. So unitialized samplers might have other first setted texture location
    constexpr size_t samplerLocationOffset = 16;

    struct FramebufferOnly {
        ID fbo = 0;
    };
    struct FullFramebuffer {
        ID fbo = 0;
        ID rbo = 0;
        ID framebufferTexture = 0;
    };

    using FramebufferCases = std::variant<FramebufferOnly, FullFramebuffer>;

    struct FramebufferInfo {
        FramebufferCases framebuffer;
        bool useStencil = true;
        bool useDepth = true;

        ID fbo() const {
            return std::visit(overload {
                [&](gl::FullFramebuffer const &frame) {
                    return frame.fbo;
                },
                [&](gl::FramebufferOnly const &frame) {
                    return frame.fbo;
                }
            }, framebuffer);
        }
    };

    FullFramebuffer inline makeFullFrameBuffer(std::pair<size_t, size_t> resolution) {
        ID fbo;
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);

        ID framebufferTexture;
        glGenTextures(1, &framebufferTexture);
        glBindTexture(GL_TEXTURE_2D, framebufferTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, resolution.first, resolution.second, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, framebufferTexture, 0);

        ID rbo;
        glGenRenderbuffers(1, &rbo);
        glBindRenderbuffer(GL_RENDERBUFFER, rbo);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, resolution.first, resolution.second);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);

        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            std::cerr << "ILLFORMED::FRAMEBUFFER" << std::endl;
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        return FullFramebuffer {
            .fbo = fbo,
            .framebufferTexture = framebufferTexture,
            .rbo = rbo
        };
    }

    FramebufferOnly inline defaultFrameBuffer() {
        return FramebufferOnly {};
    }
}