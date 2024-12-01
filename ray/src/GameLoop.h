#pragma once

#include "SDL.h"

#include "WindowController.h"

class GameLoop
{
public:
    GameLoop(GameLoop const &other) = delete;
    GameLoop &operator=(GameLoop const &other) = delete;

    inline static GameLoop &instance()
    {
        static GameLoop loop;
        return loop;
    }

    void start()
    {
        windowController.showWindow();
        while (!shouldClose)
        {
            processInput();
        }
        windowController.~WindowController();
    }

private:
    GameLoop() : windowController(WindowController({800, 600})) {}

    inline void processInput()
    {
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            switch (event.type)
            {
            case SDL_KEYDOWN:
                switch (event.key.keysym.sym)
                {
                case SDLK_UP:
                    break;
                case SDLK_DOWN:
                    break;
                }
                break;
            case SDL_QUIT:
            case SDL_WINDOWEVENT_CLOSE:
                shouldClose = true;
                break;
            default:
                break;
            }
        }
    }

    WindowController windowController;
    bool shouldClose = false;
};