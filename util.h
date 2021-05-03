#include <cuda.h>
#include <cuComplex.h>
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <sys/time.h> 
using namespace std;

typedef struct Point{
    float x, y;
}Point;

void read_file(char *filename, vector <Point> &points);

float kmeans_cpu(Point* points, Point* means,int* labels,float* dist, int iter, int n, int k, float* out);
float kmeans_gpu(Point* points, Point* cpupoints, Point* means,int* labels,float* dist,float* cpudist, int iter, int n, int k, float* out);
float kmeans_cpu_ineq(Point* points,Point* means, int* labels,float* icd,int* rid,int iter,int n,int k, float* out);
float kmeans_gpu_ineq(Point* points, Point* means, int* labels, float* icd, int* rid, int iter, int n, int k, float* out,int* m);
float kmeans_gpu_ineq_eff(Point* points,int* inds, Point* means, int* labels, float* icd, int* rid, int iter, int n, int k, float* out);
vector <vector <float>> gmix_gpu(vector <Point> points, int iter);

vector <vector <float>> gmix_cpu(vector <Point> points, int iter);

