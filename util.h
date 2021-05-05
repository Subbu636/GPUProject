#include <cuda.h>
#include <cuComplex.h>
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <sys/time.h> 
#include "cublas_v2.h"
using namespace std;

typedef struct Point{
    double x, y;
}Point;

void read_file(char *filename, vector <Point> &points);

float kmeans_cpu(Point* points, Point* means,int* labels,float* dist, int iter, int n, int k, float* out);
float kmeans_gpu(Point* points, Point* cpupoints, Point* means,int* labels,float* dist,float* cpudist, int iter, int n, int k, float* out);
float kmeans_cpu_ineq(Point* points,Point* means, int* labels,float* icd,int* rid,int iter,int n,int k, float* out);
float kmeans_gpu_ineq(Point* points, Point* means, int* labels, float* icd, int* rid, int iter, int n, int k, float* out,int* m);
float kmeans_gpu_ineq_eff(Point* points,int* inds, Point* means, int* labels, float* icd, int* rid, int iter, int n, int k, float* out);
vector <vector <float>> gmix_gpu(vector <Point> points, int iter);

vector <Point> kmeans_gpu(vector <Point> points, int iter);


// Gmix Functions
double *read_file_array(char *fname, int d,int *l);
__host__ __device__ double *create(int m, int n);
double *init_prob(int k, int l);
void gmix_cpu(double *p,double *r, int k, int iter, int l, int d, double *ctime);
void gmix_gpu(double *p,double *r, int k, int iter, int l, int d, double *gtime);
void fprint_mat(char *fname, double *prob, int l, int k);




