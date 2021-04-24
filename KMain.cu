#include <cuda.h>
#include <cuComplex.h>
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <sys/time.h> 
#include "util.h"
using namespace std;

int main(int argc,char **argv){
    int k = ;
    struct timeval t1, t2;
    char *fname = argv[1];
    vector <Point> points;
    read_file(fname,points);
    int num_points = points.size();
    Points** gpupoints; 
    Points* gpumeans;
    int* gpulabels;
    //Updating above variables
    int iter = 1000;
    // cpu
    gettimeofday(&t1, 0);
    kmeans_cpu(points,iter);
    gettimeofday(&t2, 0);
    double ct = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    printf("CPU Time taken: %.6f ms\n", ct);
    // gpu
    gettimeofday(&t1, 0);
    kmeans_gpu(gpupoints,gpumeans,gpulabels,iter,num_points,k);
    gettimeofday(&t2, 0);
    double gt = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    printf("GPU Time taken: %.6f ms\n", gt);
}


