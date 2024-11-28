#include "SDLRenderer.h"
#include "memory"

SDLRenderer::SDLRenderer(SDL_Window *window) 
    : renderer(SDL_CreateRenderer(window, -1, 0)), 
    allocator(std::allocator<uint32_t>{}) {
    int w, h;
    SDL_GetWindowSize(window, &w, &h);
    windowSize = std::make_pair(w, h);
    decltype(this->allocator) &allocator = allocator; 
    colorBuffer = {allocator.allocate(w * h), [&allocator, w, h](uint32_t * p) {
        allocator.destroy(p);
        allocator.deallocate(p, w * h);
    }};
    zBuffer = {allocator.allocate(w * h), [&allocator, w, h](uint32_t *p) {
        allocator.destroy(p);
        allocator.deallocate(p, w * h);
    }};
    renderTarget = SDL_CreateTexture(
        renderer,
        SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STREAMING,
        w,
        h
    );
}

void SDLRenderer::update() const {

}

void SDLRenderer::render() const {
    auto const &[w, h] = windowSize;
    SDL_UpdateTexture(renderTarget, nullptr, colorBuffer.get(), w * sizeof(uint32_t));
    SDL_RenderCopy(renderer, renderTarget, nullptr, nullptr);
    memset(colorBuffer.get(), 0xFF000000, w * h);
}

SDLRenderer::~SDLRenderer() {
    
}