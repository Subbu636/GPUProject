#include <cuda.h>
#include <cuComplex.h>
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <sys/time.h> 
using namespace std;

typedef struct Point{
    double x, y;
}Point;

void read_file(char *filename, vector <Point> &points);

double kmeans_cpu(Point* points, Point* means,int* labels,double* dist, int iter, int n, int k);
double kmeans_gpu(Point* points, Point* means,int* labels,double* dist, int iter, int n, int k);
double kmeans_cpu_ineq(Point* points,Point* means, int* labels,double* icd,int* rid,int iter,int n,int k);
double kmeans_gpu_ineq(Point* points, Point* means, int* labels, double* icd, int* rid, int iter, int n, int k);
vector <vector <double>> gmix_gpu(vector <Point> points, int iter);

vector <vector <double>> gmix_cpu(vector <Point> points, int iter);

