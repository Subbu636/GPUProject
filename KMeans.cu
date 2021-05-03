#include "util.h"
#include <sys/time.h> 
void dist_update(float* dist, Point* points, Point* means, int n, int k)
{
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<k;j++)
        {
            Point a = points[i];
            Point b = means[j];
            float xdist = (a.x - b.x)*(a.x - b.x);
            float ydist = (a.y - b.y)*(a.y - b.y);
            dist[i*k + j] = sqrt(xdist + ydist);
        }
    }
}
int label_update_cpu(int* labels, float* dist, int k,int n)
{
    int change = 0; 
    for(int i=0;i<n;i++)
    {
        int l = 0;
        float mindist = dist[i*k]; 
        for(int j=1;j<k;j++)
        {
            if(mindist > dist[i*k + j]){
                l = j;
                mindist = dist[i*k + j];
            }
        }
        if(labels[i]!=l) change +=1;
        labels[i] = l;
    }
    return change;
}
void centers_update_cpu(Point* means,Point* points,int* labels,int n,int k)
{
    for(int j=0;j<k;j++)
    {
        means[j].x = 0;
        means[j].y = 0;
        float num_points = 0;
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
float kmeans_cpu(Point* points, Point* means,int* labels,float* dist, int iter, int n, int k,float* out){
    
    
        struct timeval t1, t2;

     gettimeofday(&t1, 0);
    dist_update(dist,points,means,n,k);
    // printf("dist is fine\n");
    
    int ch = label_update_cpu(labels,dist,k,n);
         gettimeofday(&t2, 0);
    // printf("labels is fine\n");
    centers_update_cpu(means,points,labels,n,k);

     float ct = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    out[0] = ct;
    out[1] = ch;
    return ct;
} 

__global__ void distance_update(float* dist, Point* points, Point* means, int n, int k)
{
    // int id = (blockIdx.x*blockDim.x) + threadIdx.x;
    // int i = blockIdx.x;
    for(int i=0;i<n;i++){
        int j = threadIdx.x;
        Point a = points[i];
        Point b = means[j];
        float xdist = (a.x - b.x)*(a.x - b.x);
        float ydist = (a.y - b.y)*(a.y - b.y);
        dist[i*k + j] = sqrt(xdist + ydist);
    }
}
__global__ void label_update(int* labels,float* dist,int k,int n,int* ch)
{
    int id = (blockIdx.x*blockDim.x) + threadIdx.x;
    if(id < n){
        int l = 0;
        float mindist = dist[id*k]; 
        for(int i=1;i<k;i++)
        {
            if(mindist > dist[id*k + i]){
                l = i;
                mindist = dist[id*k + i];
            }
        }
        if(labels[id]!=l) {
            // printf("bro!!!\n");
            atomicAdd(&(ch[0]),1);
        }
        labels[id] = l;
        // if(labels[id] == l) { 
        //         printf("bro!!!\n");
        // }
    }
}
__global__ void centers_update(Point* means,Point* points,int* labels,int n,int k)
{
    int id = threadIdx.x;
    if(id<k){
        means[id].x = 0;
        means[id].y = 0;
        float num_points = 0;
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
float kmeans_gpu(Point* points, Point* cpupoints, Point* means,int* labels,float* dist,float* cpudist, int iter, int n, int k, float* out){
    // distances size n*k 
    // write kmeans gpu  
        struct timeval t1, t2;

    
     // cpu dist
    Point cpumeans[k];
    cudaMemcpy(cpumeans,means,k*sizeof(Point),cudaMemcpyDeviceToHost);

     gettimeofday(&t1, 0);
    dist_update(cpudist,cpupoints,cpumeans,n,k);
     gettimeofday(&t2, 0);

    cudaMemcpy(dist,cpudist,n*k*sizeof(float),cudaMemcpyHostToDevice);
    float a = n/1024 + 1;
    float ct = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    out[2] = ct;
    int ch[2];
    ch[0] = 0;
    int* gch; cudaMalloc(&gch,2*sizeof(int));
    cudaMemcpy(gch,ch,2*sizeof(int),cudaMemcpyHostToDevice);
     gettimeofday(&t1, 0);
    label_update<<<a,1024>>>(labels,dist,k,n,gch);
    cudaDeviceSynchronize();
    gettimeofday(&t2, 0);
    centers_update<<<1,k>>>(means,points,labels,n,k);
    cudaDeviceSynchronize();
    // gettimeofday(&t2, 0);
    ct += (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    /// change ch[0];
    out[0] = ct;
    cudaMemcpy(ch,gch,2*sizeof(int),cudaMemcpyDeviceToHost);
    out[1] = ch[0];
    return ct;
} 
void icd_update(Point* means, float* icd, int k)
{
    for(int i=0;i<k;i++)
    {
        for(int j=0;j<k;j++)
        {
            Point a = means[i];
            Point b = means[j];
            icd[i*k + j] = sqrt ( (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) );
        }
    }
}
void rid_update(float* icd,int* rid,int k){
    for(int i=0;i<k;i++)
    {
        vector< pair<float,int> > v;
        for(int j=0;j<k;j++)
        {
            v.push_back(make_pair(icd[i*k+j],j));
        }
        sort(v.begin(),v.end());
        for(int j=0;j<k;j++)
        {
            rid[i*k + j] = v[j].second;
        }
    }

}
int label_update_ineq(Point* points,int* labels,Point* means,float* icd, int* rid , int n, int k){
    int change = 0;
     for(int i=0; i<n ;i++)
     {
         int curr_cent = labels[i];
         Point p = points[i];
         Point c = means[curr_cent];
         float d = sqrt( (p.x-c.x)*(p.x - c.x) +  (p.y-c.y)*(p.y - c.y) );
         float new_d = d;
         float new_cent = curr_cent;
         for(int j=1;j<k;j++)
         {
            int cent = rid[curr_cent*k + j];
            if (icd[curr_cent*k + cent] > 2*d )
            {
                break;
            }
            Point curr_c = means[cent];
            float curr_d =  sqrt ( (p.x-curr_c.x)*(p.x-curr_c.x) + (p.y-curr_c.y)*(p.y-curr_c.y) );
            if(curr_d < new_d)
            {
                new_d = curr_d;
                new_cent = cent;
            }
         }
         if(labels[i]!=new_cent) change+=1;
         labels[i] = new_cent;
     }
     return change;
}
float kmeans_cpu_ineq(Point* points,Point* means, int* labels,float* icd,int* rid,int iter,int n,int k,float* out)
{
    struct timeval t1, t2;

    gettimeofday(&t1, 0);
    icd_update(means,icd,k);
    rid_update(icd,rid,k);
    int ch = label_update_ineq(points,labels,means,icd,rid,n,k);
    gettimeofday(&t2, 0);
    centers_update_cpu(means,points,labels,n,k);
    float ct = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    // change ch
    out[0] = ct;
    out[1] = ch;
    return ct;
}
__global__ void label_update_ineq_gpu(Point* points,int* labels,Point* means,float* icd, int* rid,int n,int k,int* ch,int* m)
{
     int t =  ceil( (float)n/ (gridDim.x * blockDim.x) );
     int s =  blockIdx.x * (n/gridDim.x) + threadIdx.x * t;
    //  printf("%d %d\n",s,s+t);
     for(int i = s;i< s+t ;i++)
     {
         if( i < n) {
            int curr_cent = labels[i];
            Point p = points[i];
            Point c = means[curr_cent];
            float d = sqrt( (p.x-c.x)*(p.x - c.x) +  (p.y-c.y)*(p.y - c.y) );
            m[i] += 1;
            float new_d = d;
            float new_cent = curr_cent;
            for(int j=1;j<k;j++)
            {
                int cent = rid[curr_cent*k + j];

                if (icd[curr_cent*k + cent] > 2*d )
                {
                   
                    break;
                }
                
                Point curr_c = means[cent];
                float curr_d = sqrt( (p.x-curr_c.x)*(p.x-curr_c.x) + (p.y-curr_c.y)*(p.y-curr_c.y) );
                m[i] += 1;
                if(curr_d < new_d)
                {
                    new_d = curr_d;
                    new_cent = cent;
                }
            }
            if(labels[i] != new_cent) { 
                // printf("bro!!!\n");
                atomicAdd(&(ch[0]),1);
            }
            labels[i] = new_cent;
            
         }
     }
}
float kmeans_gpu_ineq(Point* points, Point* means, int* labels, float* icd, int* rid, int iter, int n, int k, float* out,int* m){
    // means , rid, icd CPU
    int minGridSize, gridSize, blockSize;
    //  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,label_update_ineq, 0, 1024); 
    // Round up according to array size 
    blockSize = 1024;
    gridSize = (n + blockSize - 1) / blockSize; 

    struct timeval t1, t2;
    float* gpuicd;
    int* gpurid;
    Point* gpumeans;
    cudaMalloc(&gpuicd,k*k*sizeof(float));
    cudaMalloc(&gpurid,k*k*sizeof(int));
    cudaMalloc(&gpumeans,k*sizeof(Point));

    gettimeofday(&t1, 0);
    icd_update(means,icd,k);
    rid_update(icd,rid,k);
    gettimeofday(&t2, 0);
    float ct = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    out[2] = ct;
    cudaMemcpy(gpumeans,means,k*sizeof(Point),cudaMemcpyHostToDevice);
    cudaMemcpy(gpuicd,icd,k*k*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(gpurid,rid,k*k*sizeof(int),cudaMemcpyHostToDevice);
    int ch[2];
    ch[0] = 0;

    int* gch; cudaMalloc(&gch,2*sizeof(int));
    cudaMemcpy(gch,ch,2*sizeof(int),cudaMemcpyHostToDevice);
    gettimeofday(&t1, 0);
    label_update_ineq_gpu<<<gridSize,blockSize>>>(points,labels,gpumeans,gpuicd, gpurid,n,k,gch,m);
    cudaDeviceSynchronize();
    gettimeofday(&t2, 0);
    ct += (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    centers_update<<<1,k>>>(gpumeans,points,labels,n,k);   
    cudaDeviceSynchronize();
    cudaMemcpy(means,gpumeans,k*sizeof(Point),cudaMemcpyDeviceToHost);
    // change ch[0]
    cudaMemcpy(ch,gch,2*sizeof(int),cudaMemcpyDeviceToHost);
    out[0] = ct;
    out[1] = ch[0];
    return ct;
}
__global__ void label_update_ineq_gpu_eff(Point* points,int* inds,int* labels,Point* means,float* icd, int* rid,int n,int k,int* ch)
{
     int t =  ceil( (float)n/ (gridDim.x * blockDim.x) );
     int s =  blockIdx.x * (n/gridDim.x) + threadIdx.x * t;
    //  printf("%d %d\n",s,s+t);
     for(int m = s; m< s+t ;m++)
     {
         if( m < n) {
            int i = m;
            int curr_cent = labels[inds[m]];
            Point p = points[i];
            Point c = means[curr_cent];
            float d =sqrt( (p.x-c.x)*(p.x - c.x) +  (p.y-c.y)*(p.y - c.y) );
            float new_d = d;
            float new_cent = curr_cent;
            for(int j=1;j<k;j++)
            {
                int cent = rid[curr_cent*k + j];
                if (icd[curr_cent*k + cent] > 2*d )
                {
                    break;
                }
                Point curr_c = means[cent];
                float curr_d = sqrt( (p.x-curr_c.x)*(p.x-curr_c.x) + (p.y-curr_c.y)*(p.y-curr_c.y) );
                if(curr_d < new_d)
                {
                    new_d = curr_d;
                    new_cent = cent;
                }
            }
            if(labels[inds[m]] != new_cent) { 
                // printf("bro!!!\n");
                atomicAdd(&(ch[0]),1);
            }
            labels[inds[m]] = new_cent;
           
         }
     }
}
__global__ void centers_update_eff(Point* means,int* inds, Point* points,int* labels,int n,int k)
{
    int id = threadIdx.x;
    if(id<k){
        means[id].x = 0;
        means[id].y = 0;
        float num_points = 0;
        for(int i=0;i<n;i++){
            if(labels[inds[i]]==id)
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
float kmeans_gpu_ineq_eff(Point* points,int* inds, Point* means, int* labels, float* icd, int* rid, int iter, int n, int k, float* out){
    // means , rid, icd CPU
    int minGridSize, gridSize, blockSize;
    //  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,label_update_ineq, 0, 1024); 
    // Round up according to array size 
    blockSize = 1024;
    gridSize = (n + blockSize - 1) / blockSize; 

    struct timeval t1, t2;
    float* gpuicd;
    int* gpurid;
    Point* gpumeans;
    cudaMalloc(&gpuicd,k*k*sizeof(float));
    cudaMalloc(&gpurid,k*k*sizeof(int));
    cudaMalloc(&gpumeans,k*sizeof(Point));

    gettimeofday(&t1, 0);
    icd_update(means,icd,k);
    rid_update(icd,rid,k);
    gettimeofday(&t2, 0);
    float ct = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    
    out[2] = ct;
    cudaMemcpy(gpumeans,means,k*sizeof(Point),cudaMemcpyHostToDevice);
    cudaMemcpy(gpuicd,icd,k*k*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(gpurid,rid,k*k*sizeof(int),cudaMemcpyHostToDevice);
    
    int ch[2];
    ch[0] = 0;
    int* gch; cudaMalloc(&gch,2*sizeof(int));
    
    cudaMemcpy(gch,ch,2*sizeof(int),cudaMemcpyHostToDevice);
    gettimeofday(&t1, 0);
    label_update_ineq_gpu_eff<<<gridSize,blockSize>>>(points,inds,labels,gpumeans,gpuicd, gpurid,n,k,gch);
    cudaDeviceSynchronize();
    gettimeofday(&t2, 0);
    ct += (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;

    centers_update_eff<<<1,k>>>(gpumeans,inds,points,labels,n,k);   
    cudaDeviceSynchronize();
    cudaMemcpy(means,gpumeans,k*sizeof(Point),cudaMemcpyDeviceToHost);
    // change ch[0]
    cudaMemcpy(ch,gch,2*sizeof(int),cudaMemcpyDeviceToHost);
    out[0] = ct;
    out[1] = ch[0];
    return ct;
}