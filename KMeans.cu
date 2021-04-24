#include "util.h"

vector <Point> kmeans_cpu(vector <Point> points, int iter){
    vector <Point> means;
    // write kmeans cpu 
    return means;
} 

__global__ void distance_update(double* dist, Points* points, Points* mean)
{
    int id = (blockIdx.x*blockDim.x) + threadIdx.x;
    int i = blockIdx.x;
    int j = threadIdx.x;
    Point a = points[i];
    Point b = mean[i];
    double xdist = (a.x - b.x)*(a.x - b.x);
    double ydist = (a.y - b.y)*(a.y - b.y);
    dist[id] = xdist + ydist;
}
void kmeans_gpu(Points* points, Points* means,int* labels,double* dist, int iter, int n, int k){
    // distances size n*k 
    // write kmeans gpu  
    distance_update<<<n,k>>>(dist,points,mean);
} 