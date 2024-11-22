
class Renderer {
    virtual void update() = 0;
    virtual void render() = 0;
    virtual ~Renderer() = 0;
};

Renderer::~Renderer() {};