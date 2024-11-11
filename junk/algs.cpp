#include <iostream>
#include <vector>
#include <cassert>
#include <list>
#include <limits>

// Queue
// LinkedList
// Vector
// Map
// 

struct MaxStack {
    int pop() {
        auto val = objects.front();
        if (val == max_)
        {
            max_map[val].second--;
            auto [next_max, counts] = max_map[val];
            if (counts <= 0)
            {
                max_map.erase(val);
                max_ = next_max;
            }
        }
        objects.front();
        return val;
    }

    void push(int value) noexcept {
        if (value >= max_) {
            max_map[value].second++;
            if (value != max_) {
                max_map[value].first = max_;
            }
            max_ = value;
        }
        objects.push_front(value);
    }

    int max() {
        return max_;
    }

    std::list<int> objects;
    int max_ = std::numeric_limits<int>::lowest();
    std::unordered_map<int, std::pair<int, int>> max_map;
};


int main() {
    MaxStack stack;

    stack.push(2);
    std::cout << "max " << stack.max_ << std::endl;
    stack.push(1);
    std::cout << "max " << stack.max_ << std::endl;
    stack.push(3);
    std::cout << "max " << stack.max_ << std::endl;
    stack.push(3);
    std::cout << "max " << stack.max_ << std::endl;
    stack.pop(); // 3
    std::cout << "max " << stack.max_ << std::endl;
    stack.pop(); // 3
    std::cout << "max " << stack.max_ << std::endl;
    return 0;
}