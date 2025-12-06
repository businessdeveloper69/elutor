#include <iostream>
#include <cmath>
using namespace std;

constexpr int NV = 15;
constexpr double PI = 3.14159265358979323846;

const double TX[NV] = {0.0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075,
                       -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125};
const double TY[NV] = {0.8, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, -0.2,
                       -0.2, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5};

struct Poly {
    double px[NV], py[NV];
    
    void set(double ox, double oy, double deg) {
        double rad = deg * PI / 180.0;
        double c = cos(rad), s = sin(rad);
        for (int i = 0; i < NV; i++) {
            px[i] = TX[i] * c - TY[i] * s + ox;
            py[i] = TX[i] * s + TY[i] * c + oy;
        }
        cout << "Set poly at (" << ox << ", " << oy << "), deg=" << deg << "\n";
        cout << "  First vertex: (" << px[0] << ", " << py[0] << ")\n";
    }
};

int main() {
    Poly p1, p2;
    p1.set(0, 0, 51.81595970232997);
    p2.set(0, 0, 51.81595970232997);
    
    cout << "Same tree at same location should not overlap with itself\n";
    cout << "p1[0] = (" << p1.px[0] << ", " << p1.py[0] << ")\n";
    cout << "p2[0] = (" << p2.px[0] << ", " << p2.py[0] << ")\n";
    
    return 0;
}
