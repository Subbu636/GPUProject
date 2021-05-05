#include "util.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__host__ __device__ double *create(int m, int n){
    return (double*)malloc(sizeof(double)*n*m);
}

__host__ __device__ void init(double *v, int m, int n, double val){
    for(int i = 0;i < m;i++){
        for(int j = 0;j < n;++j){
            v[i*n+j] = val;
        }
    }
}

__host__ __device__ void copy(double *a, double *b, int m, int n){
    for(int i = 0;i < m;i++){
        for(int j = 0;j < n;++j){
            a[i*m+j] = b[i*m+j];
        }
    }
}


__host__ __device__ double *transpose(double *trans, double *v, int m, int n){
    for(int i = 0;i < m;i++){
        for(int j = 0;j < n;j++){
            trans[j*m+i] = v[i*n+j];
        }
    }
    return trans;
}

__host__ __device__ double *matsub(double *res, double *v1, double *v2, int m, int n){
    for(int i = 0;i < m;i++){
        for(int j = 0;j < n;++j){
            res[i*n+j] = v1[i*n+j]-v2[i*n+j];
        }
    }
    return res;
}

__host__ __device__ double *matmul(double *res, double *a, double *b, int m, int s, int n){
    init(res, m, n, 0.0);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < s; k++) res[i*n+j] += a[i*s+k]*b[k*n+j];
        }
    }
    return res;
}

__host__ __device__ double *matadd(double *res, double *v1, double *v2, int m, int n){
    for(int i = 0;i < m;i++){
        for(int j = 0;j < n;++j){
            res[i*n+j] = v1[i*n+j]+v2[i*n+j];
        }
    }
    return res;
}

__host__ __device__ double *matsmul(double *res, double *v, double val, int m, int n){
    for(int i = 0;i < m;i++){
        for(int j = 0;j < n;++j){
            res[i*n+j] = val*v[i*n+j];
        }
    }
    return res;
}

__host__ __device__ double norm(double *x, double *mu, double *sigma, int d){
    double sig = 1.0;
    for(int i = 0;i < d;++i){
        sig = sig * sqrt(sigma[i*d+i]);
    }
    double deno = sqrt(pow((double)(2.0*3.14159),(double)d))*sig;
    double expo = 0.0;
    for(int i = 0;i < d;++i){
        expo += ((x[i]-mu[i])*(x[i]-mu[i])/sigma[i*d+i]);
    }
    // printf("%lf %lf %lf\n",exp(-0.5*expo),deno, expo);
    return exp(-0.5*expo)/deno;
}

__host__ __device__ void matprint(double *a, int m, int n){
    for(int i = 0;i < m;i++){
        for(int j = 0;j < n;j++){
            printf("%f ",a[i*n+j]);
        }
        printf ("\n");
    }
}

void gmix_cpu(double *p,double *r, int k, int iter, int l, int d, double *ctime){

    // Debugging
    // double arr[4] = {2.0,1.0,1.0,2.0};
    // double mu[2] = {1.0, 1.0}, x[2] = {1.0, 0.0};
    // cout<<norm(x,mu,arr,2)<<endl;

    double n[k],pi[k],mu[k*d],sigma[k*d*d];
    double temp[d], var[d*d],trans[d];
    struct timeval t1, t2;

    // matprint(r,2,k);
    // cout<<"r----------------------------"<<endl;

    for(int t = 0;t < iter;++t){
        gettimeofday(&t1, 0);
        // M-step
        init(n,1,k,0.0);
        for(int i = 0;i < k;++i){
            for(int j = 0;j < l;++j){
                n[i] += r[j*k+i];
            }
        }
        // matprint(n,1,k);
        // cout<<"n--------------------------------"<<endl;
        for(int i = 0;i < k;++i){
            pi[i] = n[i]/(double)l;
        }
        // matprint(pi,1,k);
        // cout<<"pi--------------------------------"<<endl;
        init(mu,k,d,0.0);
        for(int i = 0;i < k;++i){
            for(int j = 0;j < l;++j){
                matadd(&mu[i*d],&mu[i*d],matsmul(temp,&p[j*d],r[j*k+i],d,1),d,1);
            }
        }
        for(int i = 0;i < k;++i){
            matsmul(&mu[i*d],&mu[i*d],1/n[i],d,1);
        }
        // matprint(mu,1,k*d);
        // cout<<"mu--------------------------------"<<endl;
        init(sigma,k,d*d,0.0);
        for(int i = 0;i < k;++i){
            for(int j = 0;j < l;++j){
                matsub(temp,&p[j*d],&mu[i*d],d,1);
                matmul(var,temp,transpose(trans,temp,d,1),d,1,d);
                matsmul(var,var,r[j*k+i],d,d);
                matadd(&sigma[i*d*d],&sigma[i*d*d],var,d,d);
            }
        }
        for(int i = 0;i < k;++i){
            matsmul(&sigma[i*d*d],&sigma[i*d*d],1/n[i],d,d);
        }
        // matprint(sigma,1,k*d*d);
        // cout<<"sigma--------------------------------"<<endl;
        gettimeofday(&t2, 0);
        ctime[t*2] = ((double)(1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0);
        cout<<"CPU:"<<t<<endl;
        gettimeofday(&t1, 0);
        // E-step
        for(int i = 0;i < l; ++i){
            double s = 0;
            for(int j = 0;j < k; ++j){
                r[i*k+j] = pi[j]*norm(&p[i*d],&mu[j*d],&sigma[j*d*d],d);
                s+=r[i*k+j];
            }
            assert(s != 0.0);
            for(int j = 0;j < k;++j){
                r[i*k+j] = r[i*k+j]/s;
            }
        }
        // matprint(r,1,k);
        // cout<<"r--------------------------------"<<endl;
        gettimeofday(&t2, 0);
        ctime[t*2+1] = ((double)(1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0);
    }
    return;  
} 



__global__ void compute_n_pi(double *pi, double *n, double *r, int l, int k){
    int id = (blockIdx.x*blockDim.x)+threadIdx.x;
    if(id >= k) return;
    n[id] = 0.0;
    for(int i = 0;i < l;++i){
        n[id] += r[i*k+id];
    }
    pi[id] = n[id]/(double)l;
}

__global__ void compute_pi(double *pi, double *n, int l, int k){
    int id = (blockIdx.x*blockDim.x)+threadIdx.x;
    if(id >= k) return;
    pi[id] = n[id]/(double)l; 
}

__global__ void compute_div(double *m, double *n, int k, int d){
    int id = (blockIdx.x*blockDim.x)+threadIdx.x;
    if(id >= k) return;
    for(int i = 0;i < d;i++){
        m[id*d+i] = m[id*d+i]/n[id];
    }
}

__global__ void compute_v(double *v, double *p, double *mu, double *r, int l, int k, int d){
    int id = (blockIdx.x*blockDim.x)+threadIdx.x;
    if(id >= k*l) return;
    int i = id/k, j = id%k;
    double *temp = create(d,1), *var = create(d,d), *trans = create(d,1);
    matsub(temp,&p[i*d],&mu[j*d],d,1);
    matmul(var,temp,transpose(trans,temp,d,1),d,1,d);
    for(int s = 0;s < d*d;++s){
        v[l*(j*d*d + s) + i] = var[s]*r[i*k+j];
    }
    free(temp); free(var); free(trans);
}

__global__ void compute_r(double *r, double *pi, double *p, double *mu, double *sigma, int l, int k, int d){
    int i = (blockIdx.x*blockDim.x)+threadIdx.x;
    if(i >= l) return;
    double s = 0.0;
    for(int j = 0;j < k; ++j){
        r[i*k+j] = pi[j]*norm(&p[i*d],&mu[j*d],&sigma[j*d*d],d);
        s+=r[i*k+j];
    }
    for(int j = 0;j < k;++j){
        r[i*k+j] = r[i*k+j]/s;
    }
}

__global__ void gmatprint(double *a, int m, int n){
    matprint(a,m,n);
}

void cublas_atb(double *res, const double *a, const double *b, const int m, const int k, const int n, cublasHandle_t &handle) {
    int lda=n,ldb=m,ldc=n;
    const double alf = 1.0;
    const double bet = 0.0;
    const double *alpha = &alf;
    const double *beta = &bet;
    cublasStatus_t stat;
    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, alpha, b, lda, a, ldb, beta, res, ldc);
    // printf ("%d\n",stat); 
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("#cublas multiply error\n");
        exit(1);
    }
}

void cublas_ab(double *res, const double *a, const double *b, const int m, const int k, const int n,cublasHandle_t &handle) {
    int lda=n,ldb=k,ldc=n;
    const double alf = 1.0;
    const double bet = 0.0;
    const double *alpha = &alf;
    const double *beta = &bet;
    cublasStatus_t stat;
    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha, b, lda, a, ldb, beta, res, ldc);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("#cublas multiply error\n");
        exit(1);
    }
}

__global__ void gmatmul_ab(double *res, double *a, double *b, int m, int k, int n){
    int id = (blockIdx.x*blockDim.x)+threadIdx.x;
    if(id >= m*n) return;
    int i = id/n, j = id%n;
    res[i*n+j] = 0.0;
    for(int l = 0;l < k;++l){
        res[i*n+j] += (a[i*k+l]*b[l*n+j]);
    }
}
__global__ void gmatmul_atb(double *res, double *a, double *b, int m, int k, int n){
    int id = (blockIdx.x*blockDim.x)+threadIdx.x;
    if(id >= m*n) return;
    int i = id/n, j = id%n;
    res[i*n+j] = 0.0;
    for(int l = 0;l < k;++l){
        res[i*n+j] += (a[l*m+i]*b[l*n+j]);
    }
}

__global__ void gmatmul_atb_mu(double *res, double *a, double *b, double *nn, int m, int k, int n){
    int id = (blockIdx.x*blockDim.x)+threadIdx.x;
    if(id >= m*n) return;
    int i = id/n, j = id%n;
    res[i*n+j] = 0.0;
    for(int l = 0;l < k;++l){
        res[i*n+j] += (a[l*m+i]*b[l*n+j]);
    }
    res[i*n+j] = res[i*n+j]/nn[i];
}

__global__ void gmatmul_ab_sigma(double *res, double *a, double *b, double *nn, int m, int k, int n,int d){
    int id = (blockIdx.x*blockDim.x)+threadIdx.x;
    if(id >= m*n) return;
    int i = id/n, j = id%n;
    res[i*n+j] = 0.0;
    for(int l = 0;l < k;++l){
        res[i*n+j] += (a[i*k+l]*b[l*n+j]);
    }
    res[i*n+j] = res[i*n+j]/nn[i/(d*d)];
}
 
void gmix_gpu(double *cp,double *cr, int k, int iter, int l, int d, double *gtime){
    cudaDeviceSetLimit(cudaLimitStackSize, 1 << 16);
    gpuErrchk(cudaPeekAtLastError());
    // debugging
    // double *res, *a, *b;
    // cudaMalloc(&res, 6*sizeof(double));
    // cudaMalloc(&a, 6*sizeof(double));
    // cudaMalloc(&b, 6*sizeof(double));
    // double ca[6] = {1.0,2.0,3.0,4.0,5.0,6.0}, cb[6] = {1.0,0.0,0.0,1.0,1.0,0.0};
    // cudaMemcpy(a,ca,6*sizeof(double),cudaMemcpyHostToDevice);
    // cudaMemcpy(b,cb,6*sizeof(double),cudaMemcpyHostToDevice);
    // cublas_atb(res, a, b, 3, 2, 3);
    // gmatprint<<<1,1>>>(res,3,3);
    

    double *n,*pi,*mu,*sigma;
    cudaMalloc(&n,k*sizeof(double));
    cudaMalloc(&pi,k*sizeof(double));
    cudaMalloc(&mu,k*d*sizeof(double));
    cudaMalloc(&sigma,k*d*d*sizeof(double));
    double *temp,*var,*trans,*lones;
    cudaMalloc(&temp,d*sizeof(double));
    cudaMalloc(&var,d*d*sizeof(double));
    cudaMalloc(&trans,d*sizeof(double));
    cudaMalloc(&lones,l*sizeof(double));
    double clo[l]; fill_n(clo, l, 1.0);
    cudaMemcpy(lones,clo,l*sizeof(double),cudaMemcpyHostToDevice);

    double *p,*r,*v;
    cudaMalloc(&p,l*d*sizeof(double));
    cudaMalloc(&r,l*k*sizeof(double));
    cudaMalloc(&v,l*d*d*k*sizeof(double));
    cudaMemcpy(p,cp,l*d*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(r,cr,l*k*sizeof(double),cudaMemcpyHostToDevice);
    gpuErrchk(cudaPeekAtLastError());

    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("#handle create error\n");
        exit(1);
    }
    struct timeval t1, t2;
    int blockSize = 256;

    // rik -> P(aprior), pi -> alpha, 

    for(int t = 0; t < iter;++t){
        gettimeofday(&t1, 0);
        // M-step
        // cublas_atb(n,r,lones,k,l,1,handle);
        compute_n_pi<<<(k),1>>>(pi, n, r, l, k);
        // gmatmul_atb<<<(k/blockSize)+1,blockSize>>>(n,r,lones,k,l,1);
        cudaDeviceSynchronize();
        // gmatprint<<<1,1>>>(n, 1, k);
        // cudaDeviceSynchronize();
        // cout<<"n-----------------------------"<<endl;
        // compute_pi<<<(k/blockSize)+1,blockSize>>>(pi,n,l,k);
        // cudaDeviceSynchronize();
        // gmatprint<<<1,1>>>(pi, 1, k);
        // cudaDeviceSynchronize();
        // cout<<"pi-----------------------------"<<endl;
        cublas_atb(mu, r, p, k, l, d,handle);
        // gmatmul_atb_mu<<<(k*d),1>>>(mu,r,p,n,k,l,d);
        // gmatmul_atb<<<(k*d/blockSize)+1,blockSize>>>(mu,r,p,k,l,d);
        cudaDeviceSynchronize();
        // gmatprint<<<1,1>>>(mu, 1, k*d);
        // cudaDeviceSynchronize();
        // cout<<"mu-----------------------------"<<endl;
        compute_div<<<(k/blockSize)+1,blockSize>>>(mu,n,k,d);
        cudaDeviceSynchronize();
        // gmatprint<<<1,1>>>(mu, 1, k*d);
        // cudaDeviceSynchronize();
        // cout<<"mu-----------------------------"<<endl;
        compute_v<<<(k*l/blockSize)+1,blockSize>>>(v,p,mu,r,l,k,d);
        cudaDeviceSynchronize();
        // gpuErrchk(cudaPeekAtLastError());
        // gmatprint<<<1,1>>>(v, 1, k);
        // cudaDeviceSynchronize();
        // cout<<"v-----------------------------"<<endl;
        cublas_ab(sigma, v, lones, d*d*k,l,1,handle);
        // gmatmul_ab_sigma<<<(k*d*d),1>>>(sigma,v,lones,n,d*d*k,l,1,d);
        // gmatmul_ab<<<(k*d*d/blockSize)+1,blockSize>>>(sigma,v,lones,d*d*k,l,1);
        cudaDeviceSynchronize();
        // gpuErrchk(cudaPeekAtLastError());
        // gmatprint<<<1,1>>>(sigma, k, d*d);
        // cudaDeviceSynchronize();
        // cout<<"sigma-----------------------------"<<endl;
        compute_div<<<(k/blockSize)+1,blockSize>>>(sigma,n,k,d*d);
        cudaDeviceSynchronize();
        // gpuErrchk(cudaPeekAtLastError());
        // gmatprint<<<1,1>>>(sigma, k, d*d);
        // cudaDeviceSynchronize();
        // cout<<"sigma-----------------------------"<<endl;
        gettimeofday(&t2, 0);
        gtime[t*2] = ((double)(1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0);
        gpuErrchk(cudaPeekAtLastError());
        cout<<"GPU:"<<t<<endl;
        gettimeofday(&t1, 0);
        // E-step
        // gmatprint<<<1,1>>>(r, 1, k);
        // cudaDeviceSynchronize();
        // cout<<"r-----------------------------"<<endl;
        compute_r<<<(l/blockSize)+1,blockSize>>>(r, pi, p, mu, sigma, l, k, d);
        cudaDeviceSynchronize();
        // gmatprint<<<1,1>>>(r, 1, 10);
        // cudaDeviceSynchronize();
        // cout<<"r-----------------------------"<<endl;
        gettimeofday(&t2, 0);
        gtime[t*2+1] = ((double)(1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0);
        gpuErrchk(cudaPeekAtLastError());
    }

    cudaMemcpy(cr,r,l*k*sizeof(double),cudaMemcpyDeviceToHost);

    stat = cublasDestroy(handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("#handle destroy error\n");
        exit(1);
    }
    cudaFree(n); cudaFree(pi); cudaFree(mu); cudaFree(sigma); 
    cudaFree(p); cudaFree(r); cudaFree(v);
    cudaFree(temp); cudaFree(var); cudaFree(trans); cudaFree(lones);
    gpuErrchk(cudaPeekAtLastError());
    return;
}


