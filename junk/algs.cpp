#include <iostream>
#include <vector>

using namespace std;

template<typename T>
void printAll(T &&arg) {
    cout << arg << endl;
}

template<typename T>
T check(T &&arg) {
    return arg;
}

template<typename... T>
void printAll(T&&... args) {
    printAll(std::forward<T&&...>(check(args)...));
}

vector<vector<int>> fill_spiral(int n) {
    vector<vector<int>> res = vector<vector<int>>(
        n,
        vector<int>(n, -1)
    );
    
    vector<pair<int, int>> directions = {
        {1, 0},
        {0, 1}, 
        {-1, 0},
        {0, -1}
    };
    
    int dir = 0;
    int x{0}, y{0};
    int idx = 1;
    
    while (true) {
        if (idx > n * n) { return res; }
        if (n > y && y >= 0
            && n > x && x >= 0
            && res[y][x] == -1) {
            res[y][x] = idx++;
            x += directions[dir % 4].first;
            y += directions[dir % 4].second;
        } else {
            x -= directions[dir % 4].first;
            y -= directions[dir % 4].second;
            dir++;
            x += directions[dir % 4].first;
            y += directions[dir % 4].second;
        }
    }
    
    return res;
}

int main()
{
    auto res = fill_spiral(1);
    for (auto &row : res) {
        for (auto &e : row) {
            cout << e;
        }
        cout << endl;
    }
    
    printAll(1, 2.0f, 3.0, "");
    
    return 0;
}