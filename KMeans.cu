#include "util.h"

void dist_update(double* dist, Point* points, Point* means, int n)
{
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<k;j++)
        {
            Point a = points[i];
            Point b = means[j];
            double xdist = (a.x - b.x)*(a.x - b.x);
            double ydist = (a.y - b.y)*(a.y - b.y);
            dist[i*n + j] = xdist + ydist;
        }
    }
}
void label_update_cpu(int* labels, double* dist, int k,int n)
{
    for(int i=0;i<n;i++)
    {
        int l = 0;
        double mindist = dist[i*n]; 
        for(int j=1;j<k;j++)
        {
            if(mindist > dist[i*n + j]){
                l = j;
                mindist = dist[i*n + j];
            }
        }
        labels[i] = l;
    }
}
void centers_update_cpu(Point* means,Point* points,int* labels,int n,int k)
{
    for(int j=0;j<k;j++)
    {
        means[j].x = 0;
        means[j].y = 0;
        double num_points = 0;
        for(int i=0;i<n;i++){
            if(labels[i]==j)
            {
                num_points+=1;
                means[j].x += points[i].x;
                means[j].y += points[i].y;
            }
        }
        means[j].x /= num_points;
        means[j].y /= num_points;
    }
}
void kmeans_cpu(Point* points, Point* means,int* labels,double* dist, int iter, int n, int k){
    
    // write kmeans cpu 
    dist_update(dist,points,means,n);
    label_update_cpu(labels,dist,k,n);
    centers_update_cpu(means,points,labels,n,k);
} 

__global__ void distance_update(double* dist, Point* points, Point* means)
{
    int id = (blockIdx.x*blockDim.x) + threadIdx.x;
    int i = blockIdx.x;
    int j = threadIdx.x;
    Point a = points[i];
    Point b = means[j];
    double xdist = (a.x - b.x)*(a.x - b.x);
    double ydist = (a.y - b.y)*(a.y - b.y);
    dist[id] = xdist + ydist;
}
__global__ void label_update(int* labels,double* dist,int k,int n)
{
    int id = threadIdx.x;
    if(id < n){
        int l = 0;
        double mindist = dist[id*n]; 
        for(int i=1;i<k;i++)
        {
            if(mindist > dist[id*n + i]){
                l = i;
                mindist = dist[id*n + i];
            }
        }
        labels[id] = l;
    }
}
__global__ void centers_update(Point* means,Point* points,int* labels,int n,int k)
{
    int id = threadIdx.x;
    if(id<k){
        means[id].x = 0;
        means[id].y = 0;
        double num_points = 0;
        for(int i=0;i<n;i++){
            if(labels[i]==id)
            {
                num_points+=1;
                means[id].x += points[i].x;
                means[id].y += points[i].y;
            }
        }
        means[id].x /= num_points;
        means[id].y /= num_points;
    }
}
void kmeans_gpu(Point* points, Point* means,int* labels,double* dist, int iter, int n, int k){
    // distances size n*k 
    // write kmeans gpu  
    distance_update<<<n,k>>>(dist,points,mean);
    label_update<<<1,n>>>(labels,dist,k,n);
    centers_update<<<1,k>>>(means,points,labels,n,k);
} 