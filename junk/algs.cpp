#include <iostream>
#include <vector>
#include <unordered_map>
#include <cassert>
#include <list>
#include <limits>

using namespace std;

vector<pair<int, int>> interval_intersections(vector<pair<int, int>> a, vector<pair<int, int>> b) {
    vector<pair<int, int>> res;
    size_t left{0};
    size_t right{0};
    pair<int, int> current_intersection;
    while (left < a.size() && right < b.size()) {
        auto const &[a0, a1] = a[left];
        auto const &[b0, b1] = b[right];

        // There're four positive cases
        // [b0, b1] inside [a0, a1] increment right
        // [a0, a1] inside [b0, b1] increment left
        // b0 < [a1] < b1  ~> [b0, a1] is intersection increment left
        // a0 < b1 < a1 ~> [a0, b1] is intersection increment right
        if (a0 <= b0 && b0 <= a1 && a0 <= b1 && b1 <= a1) {
            res.push_back(make_pair(b0, b1));
            right++;
        } else if (b0 <= a0 && a0 <= b1 && b0 <= a1 && a1 <= b1) {
            res.push_back(make_pair(a0, a1));
            left++;
        } else if (b0 <= a1 && a1 <= b1) {
            res.push_back(make_pair(b0, a1));
            left++;
        } else if (a0 <= b1 && b1 <= a1) {
            res.push_back(make_pair(a0, b1));
            right++;
        } else if (a1 < b0) {
            left++;
        } else if (b1 < a0) {
            right++;
        }
    }

    vector<pair<int, int>> final_result{res[0]};
    for (size_t i = 1; i < res.size(); i++) {
        if (res[i].first == final_result.back().second) {
            final_result.back().second = res[i].second;
        } else {
            final_result.push_back(res[i]);
        }
    }
    return final_result;
}

int main()
{
    auto res = interval_intersections({make_pair(8, 12), make_pair(17, 22)}, {make_pair(5, 11), make_pair(14, 18), make_pair(20, 23), make_pair(42, 55)});

    for (auto const & p: res) {
        cout << p.first << " " << p.second << endl;
    }

    auto res2 = interval_intersections({make_pair(9, 15), make_pair(18, 21)}, {make_pair(10, 14), make_pair(21, 22)});

    for (auto const &p : res2)
    {
        cout << p.first << " " << p.second << endl;
    }
    return 0;
}