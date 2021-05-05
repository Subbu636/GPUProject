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

double kmeans_cpu(Point* points, Point* means,int* labels,double* dist, int iter, int n, int k);
double kmeans_gpu(Point* points, Point* cpupoints, Point* means,int* labels,double* dist,double* cpudist, int iter, int n, int k);
double kmeans_cpu_ineq(Point* points,Point* means, int* labels,double* icd,int* rid,int iter,int n,int k);
double kmeans_gpu_ineq(Point* points, Point* means, int* labels, double* icd, int* rid, int iter, int n, int k);

vector <Point> kmeans_gpu(vector <Point> points, int iter);


// Gmix Functions
double *read_file_array(char *fname, int d,int *l);
__host__ __device__ double *create(int m, int n);
double *init_prob(int k, int l);
void gmix_cpu(double *p,double *r, int k, int iter, int l, int d, double *ctime);
void gmix_gpu(double *p,double *r, int k, int iter, int l, int d, double *gtime);
void fprint_mat(char *fname, double *prob, int l, int k);




