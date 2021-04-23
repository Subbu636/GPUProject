#include <cuda.h>
#include <cuComplex.h>
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <sys/time.h> 
using namespace std;

class Point{
    public:
    double x, y;
    double distance(Point p);
};

void read_file(char *filename, vector <Point> &points);

vector <Point> kmeans_cpu(vector <Point> points, int iter);

vector <Point> kmeans_gpu(vector <Point> points, int iter);

vector <vector <double>> gmix_gpu(vector <Point> points, int iter);

vector <vector <double>> gmix_cpu(vector <Point> points, int iter);

