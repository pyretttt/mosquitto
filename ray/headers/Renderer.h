
class Renderer {
public:
    virtual void update() const = 0;
    virtual void render() const = 0;
    virtual ~Renderer() = 0;
};

Renderer::~Renderer() {};