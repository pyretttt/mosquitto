#include <utility>
#include <functional>

struct SDL_Window;

struct WindowController {
    WindowController(std::pair<int, int> size);
    
    void showWindow() const;
    ~WindowController();
private:
    std::function<void()> windowInit;
    SDL_Window *window;
};