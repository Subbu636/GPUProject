#include <cuda.h>
#include <cuComplex.h>
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <sys/time.h> 
#include "util.h"
using namespace std;
Points* read(char *filename){
    FILE *file;
    file = fopen(filename, "r");
    if (!file)  {
        printf("#cannot open input file");
        return;
    }
    vector<Point> points;
    double x,y;
    int n = 0;
    while(fscanf(file,"%lf %lf",&x,&y) != EOF){
        Point p;
        p.x = x;
        p.y = y;
        points.push_back(p);
        n++;
    }
     Point* ps = (Point*) malloc(n*sizeof(Point));
     for(int i=0;i<n;i++)
     {
         ps[i] = points[i];
     }
     return ps;
}
int main(int argc,char **argv){
    int seed = 1234;
    struct timeval t1, t2;
    char *fname = argv[1];
     int k = ;
    Point* points;
    points = read(fname);
    int num_points = (int)( sizeof(points) / sizeof(points[0]) );
    Point* gpupoints;
    cudaMalloc(&gpupoints, num_points*sizeof(Point));
    // Point* cpoints;
    // cpoints = (Point*) malloc(n*sizeof(Point));
    cudaMemcpy(gpupoints,points,num_points*sizeof(Point),cudaMemcpyHostToDevice); 
    Point* means = (Point*) malloc(k*sizeof(Point));
    int* labels = (int*) malloc(num_points*sizeof(int));
    int num[k];
    for(int i=0;i<k;i++) { means[i] = 0; num[i] = 0;}
    
    for(int i=0;i<num_points;i++)
    {
        labels[i] = (srand(seed))%k;
    }
    for(int i=0;i<num_points;i++)
    {
        num[labels[i]] ++;
        means[labels[i]].x += points[i].x;
        means[labels[i]].y += points[i].y;
    }
    for(int i=0;i<k;i++)
    {
        means[i].x /= (double) num[i];
         means[i].y /= (double) num[i];
    }
    Points* gpumeans;
    cudaMalloc(&gpumeans, k*sizeof(Point));
    cudaMemcpy(gpumeans,means,k*sizeof(Point),cudaMemcpyHostToDevice); 
    int* gpulabels;
    cudaMalloc(&gpulabels, num_points*sizeof(int));
    cudaMemcpy(gpulabels,labels,num_points*sizeof(int),cudaMemcpyHostToDevice); 
    //Updating above variables
    int iter = 1000;
    // cpu
    double dist[num_points*k];
    double* gpudist;
    cudaMalloc(&gpudist, num_points*k*sizeof(double));
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


