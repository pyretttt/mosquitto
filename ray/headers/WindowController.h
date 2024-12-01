#include <utility>
#include <functional>

struct SDL_Window;

struct WindowController {
    WindowController(std::pair<int, int> size);
    ~WindowController();
    
    void showWindow() const;
private:
    std::function<void()> windowInit;
    SDL_Window *window;
};