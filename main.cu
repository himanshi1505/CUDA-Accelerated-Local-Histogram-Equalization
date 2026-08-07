#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <climits>
#include <algorithm>
#include <omp.h>
#include <cuda_runtime.h>

#define ll long long
#define rep(i,a,b ) for(int i=(a); i<(b); i++)
#define repl(i,a,b ) for(ll i=(a); i<(b); i++)

static inline void verify_cuda(cudaError_t result, const char* file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA failed at %s:%d - %s\n",
                file, line, cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(call) verify_cuda((call), __FILE__, __LINE__)

#define BLOCK_SIZE 256
#define MAX_K 128

__device__ __forceinline__
bool is_worse(ll da, int ia, ll db, int ib,const int* __restrict__ px, const int* __restrict__ py,   const int* __restrict__ pz) {
    if (da != db) return da > db;
    int ax = px[ia], bx = px[ib];
    if (ax != bx) return ax > bx;
    int ay = py[ia], by = py[ib];
    if (ay != by) return ay > by;
    return pz[ia] > pz[ib];
}

__device__ __forceinline__
void heap_swap(ll *hd, int *hi, int a, int b) {
    ll dtmp = hd[a]; hd[a] = hd[b];hd[b] = dtmp; int itmp = hi[a];hi[a] = hi[b];hi[b] = itmp;
}

__device__ __forceinline__
void heap_sift_down(ll *hd, int *hi, int sz, int root,  const int* __restrict__ px,    const int* __restrict__ py,       const int* __restrict__ pz) {
    int cur = root;

    while (true) {
        int left = (cur << 1) + 1;
        int right = left + 1;
        int worst = cur;

        if (left < sz && is_worse(hd[left], hi[left], hd[worst], hi[worst], px, py, pz)) {  worst = left; }
        if (right < sz &&is_worse(hd[right], hi[right], hd[worst], hi[worst], px, py, pz)) {worst = right;}

        if (worst == cur) break;

        heap_swap(hd, hi, cur, worst);
        cur = worst;
    }
}

__device__ __forceinline__
void heap_sift_up(ll *hd, int *hi, int pos, const int* __restrict__ px, const int* __restrict__ py, const int* __restrict__ pz) {
    int cur = pos;

    while (cur > 0) {
        int parent = (cur - 1) >> 1;

        if (!is_worse(hd[cur], hi[cur], hd[parent], hi[parent], px, py, pz)) {
            break;
        }

        heap_swap(hd, hi, cur, parent);
        cur = parent;
    }
}

__device__ __forceinline__
void heap_push(ll *hd, int *hi, int &hs, int cap,ll d, int idx, const int* __restrict__ px, const int* __restrict__ py, const int* __restrict__ pz) {
    if (hs < cap) {
        int pos = hs;hd[pos] = d;hi[pos] = idx; ++hs;
        heap_sift_up(hd, hi, pos, px, py, pz);
        return;
    }
     if (is_worse(hd[0], hi[0], d, idx, px, py, pz)) { hi[0] = idx;hd[0] = d;  heap_sift_down(hd, hi, cap, 0, px, py, pz); }
}

__device__ __forceinline__ 
void knn_equalize_histogram(const int* __restrict__ pI, int* nbr, int nbr_cnt, int qI, int m, int tid, int* __restrict__ out) {
    int min_intensity = qI;
    int count_leq_qI = 1;
    
    #pragma unroll 8
    for (int i = 0; i < nbr_cnt; i++) {
        int I = __ldg(&pI[nbr[i]]);
        if (I < min_intensity) min_intensity = I;
        if (I <= qI) count_leq_qI++;
    }
    
    int cdf_min = (qI == min_intensity) ? 1 : 0;
    
    #pragma unroll 8
    for (int i = 0; i < nbr_cnt; i++) {
        if (__ldg(&pI[nbr[i]]) == min_intensity) cdf_min++;
    }
    
    if (m == cdf_min) {
        out[tid] = qI;
        return;
    }
    
    double temp = ((double)(count_leq_qI - cdf_min) * 255.0) / (double)(m - cdf_min);
    int ni = (int)temp;
    
    out[tid] = (ni < 0) ? 0 : ((ni > 255) ? 255 : ni);
}

__global__ void near_neigh_gpu_calc(const int* __restrict__ px,  const int* __restrict__ py,   const int* __restrict__ pz,         const int* __restrict__ pI,  int n, int k, int* __restrict__ out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = (tid < n);

    int qx = 0, qy = 0, qz = 0, qI = 0;
    if (active) {
        qz = __ldg(&pz[tid]); qI = __ldg(&pI[tid]); qx = __ldg(&px[tid]);qy = __ldg(&py[tid]);  
        
    }

    ll hd[MAX_K]; int hi[MAX_K];  
    int hs = 0;  int cap = (k < n - 1) ? k : (n - 1);   
    __shared__ int sx[1024];     __shared__ int sy[1024];  __shared__ int sz[1024];
    int number_of_tiles = (n + 1024 - 1) / 1024;

    rep(t, 0, number_of_tiles) {
        int base = t * 1024;

        for (int offset = threadIdx.x; offset < 1024; offset += blockDim.x) {
            int index_l = base + offset;
            if (index_l < n) {
                sz[offset] = __ldg(&pz[index_l]); sy[offset] = __ldg(&py[index_l]); sx[offset] = __ldg(&px[index_l]);
               
            }
        }
        __syncthreads();

        if (active) {
            ll heapMax = (hs >= cap) ? hd[0] : LLONG_MAX;
            int end = min(1024, n - base);
            #pragma unroll 4
            for (int j = 0; j < end; ++j) {
                if (base + j == tid) 
                continue;

                ll dx = (ll)(qx - sx[j]),dy = (ll)(qy - sy[j]), dz = (ll)(qz - sz[j]);
                ll d = dx * dx + dy * dy + dz * dz;
                if (d > heapMax)
                 continue;
                heap_push(hd, hi, hs, cap, d, base + j, px, py, pz);

                if (hs >= cap) heapMax = hd[0];
            }
        }

        __syncthreads();
    }

    if (active) {
        knn_equalize_histogram(pI, hi, hs, qI, hs + 1, tid, out);
    }
}
__global__ void approx_near_neigh_gpu_calc(int n,int k,const int *__restrict__ px,const int *__restrict__ py,const int *__restrict__ pz,int minX,int minY,int minZ,int gridDimX,int gridDimY,int gridDimZ,const int *__restrict__ sortedIdx,const int *__restrict__ cellStart,const int *__restrict__ cellEnd,const int *__restrict__ pI,int cellSize,int maxShells,int *out){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;if (tid >= n) return;

    ll good_distance[MAX_K]; int good_index[MAX_K];  int good_cnt = 0;
    int limit = (k < n - 1) ? k : (n - 1);
   
    const int qx = __ldg(&px[tid]), qy = __ldg(&py[tid]), qz = __ldg(&pz[tid]);
    // const int qLabel = __ldg(&pI[tid]);

    int gx = (qx - minX) / cellSize, gy = (qy - minY) / cellSize, gz = (qz - minZ) / cellSize;
    gy = max(0, min(gy, gridDimY - 1));gz = max(0, min(gz, gridDimZ - 1));gx = max(0, min(gx, gridDimX - 1));
    
    int lastX0 = gx, lastX1 = gx,lastY0 = gy, lastY1 = gy, lastZ0 = gz, lastZ1 = gz,  shellLimit = maxShells, shell=0;


    while (shell <= shellLimit) {
        ll currentWorst = (good_cnt >= limit) ? good_distance[0] : LLONG_MAX;
        int z0 = max(0, gz - shell), z1 = min(gridDimZ - 1, gz + shell), x0 = max(0, gx - shell), x1 = min(gridDimX - 1, gx + shell), y0 = max(0, gy - shell), y1 = min(gridDimY - 1, gy + shell) ;
        rep(x,x0,x1+1)
        {
            rep(y,y0,y1+1)
            {
                rep(z,z0,z1+1)
                {
                    if (shell > 0 &&
                        x >= lastX0 && x <= lastX1 &&
                        y >= lastY0 && y <= lastY1 &&
                        z >= lastZ0 && z <= lastZ1) {
                        continue;
                    }

                    int id_of_cell = x * gridDimY * gridDimZ + y * gridDimZ + z;
                    int begin = __ldg(&cellStart[id_of_cell]);
                    int end   = __ldg(&cellEnd[id_of_cell]);

                    rep(pos,begin,end){
                        int j = __ldg(&sortedIdx[pos]);
                        if (j == tid) continue;

                        ll dz = (ll)qz - (ll)__ldg(&pz[j]), dx = (ll)qx - (ll)__ldg(&px[j]), dy = (ll)qy - (ll)__ldg(&py[j]);
                      
                        ll dist = dx * dx + dy * dy + dz * dz;

                        if (dist >= currentWorst) continue;

                        heap_push(good_distance, good_index, good_cnt, limit, dist, j, px, py, pz);

                        if (good_cnt >= limit) {
                            currentWorst = good_distance[0];
                        }
                    }
                }
            }
        }

        lastZ0 = z0; lastZ1 = z1;lastX0 = x0; lastX1 = x1; lastY0 = y0; lastY1 = y1;
        
        

        bool canStop = false;

        if (good_cnt >= limit)
         {
            if (shell < shellLimit)
             {
                ll kthDist = good_distance[0];

                ll boxZ1 = (ll)minZ + (ll)(z1 + 1) * cellSize, boxZ0 = (ll)minZ + (ll)z0 * cellSize ,boxY1 = (ll)minY + (ll)(y1 + 1) * cellSize, boxX0 = (ll)minX + (ll)x0 * cellSize,boxX1 = (ll)minX + (ll)(x1 + 1) * cellSize, boxY0 = (ll)minY + (ll)y0 * cellSize;

                ll minDx = 0, minDy = 0, minDz = 0;

                if ((ll)qx >= boxX0 && (ll)qx < boxX1) 
                {
                    ll leftGap = (ll)qx - boxX0;
                    ll rightGap = boxX1 - (ll)qx;
                    minDx = min(leftGap, rightGap);
                }

                if ((ll)qy >= boxY0 && (ll)qy < boxY1)
                 {
                    ll leftGap = (ll)qy - boxY0;
                    ll rightGap = boxY1 - (ll)qy;
                    minDy = min(leftGap, rightGap);
                }

                if ((ll)qz >= boxZ0 && (ll)qz < boxZ1)
                 {
                    ll leftGap = (ll)qz - boxZ0;
                    ll rightGap = boxZ1 - (ll)qz;
                    minDz = min(leftGap, rightGap);
                }

                ll lowerBound = min(minDx, min(minDy, minDz));
                if (kthDist < lowerBound * lowerBound) 
                {
                    canStop = true;
                }
            } 
            else 
            {
                canStop = true;
            }
        } 
        else if (shell == shellLimit)
         {
            if (good_cnt < limit && shellLimit < 8) {
                ++shellLimit;
            } 
            else {
                canStop = true;
            }
        }

        if (canStop) break;
        ++shell;
    }

    knn_equalize_histogram(pI, good_index, good_cnt, __ldg(&pI[tid]), good_cnt + 1, tid, out);
}

__global__ void gpu_summation_km(int n,int k,const int *__restrict__ py,const int *__restrict__ pz,const int *__restrict__ px,const int *__restrict__ asgns,int *y_summation,int *z_summation,int *x_summation,int *counts){
    int tid = blockIdx.x * blockDim.x + threadIdx.x; if (tid >= n) return;int temp_var= asgns[tid];atomicAdd(&z_summation[temp_var], pz[tid]);atomicAdd(&y_summation[temp_var], py[tid]); atomicAdd(&x_summation[temp_var], px[tid]); atomicAdd(&counts[temp_var], 1);}


__global__ void allocate_km_gpu_calc(int n,int k,const int *__restrict__ px,const int *__restrict__ py,const int *__restrict__ pz,const int *__restrict__ cx,const int *__restrict__ cy,const int *__restrict__ cz,
int *asgns,int *changed) {
    extern __shared__ int shared_cen[];
    int *shared_coor_x = shared_cen, *shared_coor_y = shared_cen + k, *shared_coor_z = shared_cen + 2 * k;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    ll good_dist = LLONG_MAX;

    for (int c = threadIdx.x; c < k; c += blockDim.x) {  shared_coor_z[c] = cz[c];  shared_coor_y[c] = cy[c]; shared_coor_x[c] = cx[c]; }
    __syncthreads();


    if (tid >= n) return;

    int coor_x = px[tid];

    int coor_y = py[tid];

    int good_id = 0;
    int coor_z = pz[tid];


    rep (c,0, k) {
        ll dx = (ll)coor_x - shared_coor_x[c],dy = (ll)coor_y - shared_coor_y[c],dz = (ll)coor_z - shared_coor_z[c];
        ll d = dx * dx + dy * dy + dz * dz;

        if (d < good_dist) 
        {
            good_dist = d;
            good_id = c;
        }
        else if (d == good_dist)
         {
            if (shared_coor_x[c] < shared_coor_x[good_id] ||
                (shared_coor_x[c] == shared_coor_x[good_id] && shared_coor_y[c] < shared_coor_y[good_id]) ||
                (shared_coor_x[c] == shared_coor_x[good_id] &&
                     shared_coor_y[c] == shared_coor_y[good_id] && 
                     shared_coor_z[c] < shared_coor_z[good_id])) 
                     {
                good_id = c;
            }
        }
    }

    if (asgns[tid] != good_id) {
        asgns[tid] = good_id;
        atomicOr(changed, 1);
    }
}



__global__ void make_equal_km_histogram(int n,const int *__restrict__ pI,const int *__restrict__ asg_ns,const int *__restrict__ size_of_cluster,const int *__restrict__ histogram_of_cluster,int *__restrict__ out_intensity) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; if (tid >= n) return;

    int cl = __ldg(&asg_ns[tid]);
    int m = __ldg(&size_of_cluster[cl]), intensity = __ldg(&pI[tid]);
    const int *hist = &histogram_of_cluster[cl * 256];
    int cdf_intensity = 0, cdf_min = -1;

    if (__ldg(&hist[intensity]) == m) 
    {
        out_intensity[tid] = intensity;
        return;
    }

    #pragma unroll 8
    for (int v = 0; v <= intensity; ++v) 
    {
        int h = __ldg(&hist[v]);
        cdf_intensity += h;
        
        if (cdf_intensity > 0 && cdf_min == -1) 
        {
            cdf_min = cdf_intensity;
        }
    }
    
    if (m == cdf_min)
    {
        out_intensity[tid] = intensity;
        return;
    }
    
    double temp = ((double)(cdf_intensity - cdf_min) * 255.0) / (double)(m - cdf_min);
    int newI = (int)temp;
    
    out_intensity[tid] = (newI < 0) ? 0 : ((newI > 255) ? 255 : newI);
}

static inline void write_output_file(const char* filename,
                                    const int* hx, const int* hy, const int* hz,
                                    const int* hout, int n) {
    FILE *f = fopen(filename, "w");
    if (!f)
     {
        fprintf(stderr, "Cannot open %s for writing\n", filename);
        exit(EXIT_FAILURE);
    }
    rep(i,0,n) 
    {
        fprintf(f, "%d %d %d %d\n", hx[i], hy[i], hz[i], hout[i]);
    }
    fclose(f);
}


void run_knn(int *hx, int *hy, int *hz, int *hI,
             int *dx, int *dy, int *dz, int *dI,
             int n, int k)
{
    int *dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dOut, n * sizeof(int)));

    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    near_neigh_gpu_calc<<<grid_size, BLOCK_SIZE>>>(dx, dy, dz, dI, n, k, dOut);
    CUDA_CHECK(cudaDeviceSynchronize());

    int *host_Out = (int *)malloc(n * sizeof(int));
    if (!host_Out)
     {
        printf("Memory allocation failed\n");
        exit(1);    
    }
    CUDA_CHECK(cudaMemcpy(host_Out, dOut, n * sizeof(int), cudaMemcpyDeviceToHost));

    write_output_file("knn.txt", hx, hy, hz, host_Out, n);

    free(host_Out);
    CUDA_CHECK(cudaFree(dOut));
}

__global__ void kmeans_build_histogram(int n,int k,const int *__restrict__ asg_ns,const int *__restrict__ pI,int *histogram_of_cluster,int *size_of_cluster){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    int cluster_num = __ldg(&asg_ns[i]);

    atomicAdd(&size_of_cluster[cluster_num], 1);
    atomicAdd(&histogram_of_cluster[cluster_num * 256 + __ldg(&pI[i])], 1);
}

void run_approx_knn(int *hx, int *hy, int *hz, int *hI,
    int *dx, int *dy, int *dz, int *dI,
                    int n, int k)
{
   
    int minX = hx[0], maxX = hx[0], minY = hy[0], maxY = hy[0], minZ = hz[0], maxZ = hz[0];
   
   #pragma omp parallel for reduction(max:maxX,maxY,maxZ) reduction(min:minX,minY,minZ) 
    for (int i = 1; i < n; i++) {
        minX = min(minX, hx[i]);
        minY = min(minY, hy[i]);
        minZ = min(minZ, hz[i]);

        maxX = max(maxX, hx[i]);
        maxY = max(maxY, hy[i]);
        maxZ = max(maxZ, hz[i]);
    }

    double vol = (double)((maxX - minX) + 1.0) * ((maxY - minY) + 1.0) * ((maxZ - minZ) + 1.0);

    int shellRadius = 2;
    int span = (2 * shellRadius + 1);
    span = span * span * span;

    int num_of_cells_tgt = max(1, (k * 8 + span - 1) / span);
    num_of_cells_tgt = max(1,(n + num_of_cells_tgt - 1) / num_of_cells_tgt);
    double cellVol = vol / (double)num_of_cells_tgt;
    int cellSize = max(1,(int)ceil(cbrt(cellVol)));

    int y_g = (int)((maxY - minY) / cellSize), z_g = (int)((maxZ - minZ) / cellSize), x_g = (int)((maxX - minX) / cellSize);
    y_g++; z_g++;x_g++;
    while ((ll)x_g * y_g * z_g > 8000000)
     {
        cellSize *= 2;
        x_g = (int)((maxX - minX) / cellSize) + 1;
        y_g = (int)((maxY - minY) / cellSize) + 1;
        z_g = (int)((maxZ - minZ) / cellSize) + 1;
    }
    
    ll total = 1LL * x_g * y_g * z_g;
    
   
    int *id_of_cell = (int *)malloc(n * sizeof(int));
    if (!id_of_cell )
     {
        printf("Memory allocation failed\n");
        exit(1);
    }
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int x_pt = max(0, min((hx[i] - minX) / cellSize, x_g - 1));
        int y_pt = max(0, min((hy[i] - minY) / cellSize, y_g - 1));
        int z_pt = max(0, min((hz[i] - minZ) / cellSize, z_g - 1));

        id_of_cell[i] = x_pt * y_g * z_g + y_pt * z_g + z_pt;
    }

    int *H_cnt = (int *)calloc((size_t)total, sizeof(int));   
    int *H_start = (int *)malloc((size_t)total * sizeof(int));
    int *H_end   = (int *)malloc((size_t)total * sizeof(int));

    if (!H_cnt || !H_start || !H_end )
     {
        printf("Memory allocation failed\n");
        exit(1);
    }

    rep(i, 0, n) H_cnt[id_of_cell[i]]++;
    H_start[0] = 0;
    repl(c, 1, total) H_start[c] = H_start[c - 1] + H_cnt[c - 1];
    repl(c, 0, total) H_end[c] = H_start[c] + H_cnt[c];

    int *cur = (int *)calloc((size_t)total, sizeof(int));
    int *ord = (int *)malloc(n * sizeof(int));
    if (!cur || !ord) 
    {
        printf("Memory allocation failed\n");
        exit(1);
    }
    
    rep(i, 0, n)
     {
        int c = id_of_cell[i];
        int pos = H_start[c] + cur[c];
        ord[pos] = i;
        cur[c]++;
    }

    free(id_of_cell);
    free(H_cnt);
    free(cur);

    int *dStart = nullptr; 
    CUDA_CHECK(cudaMalloc(&dStart, total * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dStart, H_start, total * sizeof(int), cudaMemcpyHostToDevice));
    free(H_start);

    int *Ordered_d = nullptr;
    CUDA_CHECK(cudaMalloc(&Ordered_d, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(Ordered_d, ord, n * sizeof(int), cudaMemcpyHostToDevice));
    free(ord);


    int *dEnd = nullptr;
    CUDA_CHECK(cudaMalloc(&dEnd, total * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dEnd, H_end, total * sizeof(int), cudaMemcpyHostToDevice));
    free(H_end);

    int *dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dOut, n * sizeof(int)));
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    approx_near_neigh_gpu_calc<<<grid_size, BLOCK_SIZE>>>( n, k, dx, dy, dz, minX, minY, minZ, x_g, y_g, z_g, Ordered_d, dStart, dEnd,dI,cellSize, shellRadius,dOut);
    CUDA_CHECK(cudaDeviceSynchronize());

    int *hout = (int *)malloc(n * sizeof(int));
    if (!hout )
    {
        printf("Memory allocation failed\n");
        exit(1);
    }
    CUDA_CHECK(cudaMemcpy(hout, dOut, n * sizeof(int), cudaMemcpyDeviceToHost));

    write_output_file("approx_knn.txt", hx, hy, hz, hout, n);

    free(hout);
    CUDA_CHECK(cudaFree(Ordered_d));
    CUDA_CHECK(cudaFree(dStart));
    CUDA_CHECK(cudaFree(dEnd));
    CUDA_CHECK(cudaFree(dOut));
}

__global__ void kmeans_gpu_reassign(const int *counts,int k, int *coor_y,int *coor_z,int *coor_x,const int *y_summation,const int *z_summation,const int *x_summation) {
    int tid= blockIdx.x * blockDim.x + threadIdx.x;
    if (tid>= k|| counts[tid] <=0) return;
    coor_y[tid] = y_summation[tid] / counts[tid]; coor_x[tid] = x_summation[tid] / counts[tid];  coor_z[tid] = z_summation[tid] / counts[tid];
    
}

void run_kmeans(int *host_x, int *host_y, int *host_z, int *host_I,
                int *gpu_x, int *gpu_y, int *gpu_z, int *gpu_I,
                int n, int k, int T)
{
    int *gpu_centroid_x = nullptr, *gpu_centroid_y = nullptr, *gpu_centroid_z = nullptr, *gpu_assignments = nullptr, *gpu_changed_flag = nullptr;
    CUDA_CHECK(cudaMalloc(&gpu_assignments, n * sizeof(int)));
    CUDA_CHECK(cudaMemset(gpu_assignments, 0xFF, n * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&gpu_centroid_x, k * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(gpu_centroid_x, host_x, k * sizeof(int), cudaMemcpyHostToDevice));

    int *gpu_sum_x = nullptr, *gpu_sum_y = nullptr, *gpu_sum_z = nullptr, *gpu_count = nullptr, *gpu_cluster_size = nullptr, *gpu_histogram = nullptr, *gpu_output = nullptr;
   
    CUDA_CHECK(cudaMalloc(&gpu_centroid_y, k * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(gpu_centroid_y, host_y, k * sizeof(int), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMalloc(&gpu_cluster_size, k * sizeof(int))); CUDA_CHECK(cudaMalloc(&gpu_sum_y, k * sizeof(int)));CUDA_CHECK(cudaMalloc(&gpu_output, n * sizeof(int)));  CUDA_CHECK(cudaMalloc(&gpu_changed_flag, sizeof(int)));  CUDA_CHECK(cudaMalloc(&gpu_histogram, k * 256 * sizeof(int))); CUDA_CHECK(cudaMalloc(&gpu_sum_z, k * sizeof(int))); CUDA_CHECK(cudaMalloc(&gpu_centroid_z, k * sizeof(int)));CUDA_CHECK(cudaMemcpy(gpu_centroid_z, host_z, k * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&gpu_sum_x, k * sizeof(int))); CUDA_CHECK(cudaMalloc(&gpu_count, k * sizeof(int)));
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE,  kgrid_size = (k + BLOCK_SIZE - 1) / BLOCK_SIZE, sharedMem = 3 * k * sizeof(int);

    rep(it,0,T)
    {
        int flag = 0;
        CUDA_CHECK(cudaMemcpy(gpu_changed_flag, &flag, sizeof(int), cudaMemcpyHostToDevice));

        allocate_km_gpu_calc<<<grid_size,BLOCK_SIZE,sharedMem>>>( n,k,gpu_x,gpu_y,gpu_z,gpu_centroid_x,gpu_centroid_y,gpu_centroid_z,gpu_assignments,gpu_changed_flag);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        int changed = 0;
        CUDA_CHECK(cudaMemcpy(&changed, gpu_changed_flag, sizeof(int), cudaMemcpyDeviceToHost));
        if (!changed)
        {
            break;
        } 

         CUDA_CHECK(cudaMemset(gpu_count, 0, k * sizeof(int))); CUDA_CHECK(cudaMemset(gpu_sum_x, 0, k * sizeof(int))); CUDA_CHECK(cudaMemset(gpu_sum_y, 0, k * sizeof(int))); CUDA_CHECK(cudaMemset(gpu_sum_z, 0, k * sizeof(int)));
        gpu_summation_km<<<grid_size,BLOCK_SIZE>>>( n,k,gpu_y,gpu_z,gpu_x,gpu_assignments,gpu_sum_y,gpu_sum_z,gpu_sum_x,gpu_count);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        kmeans_gpu_reassign<<<kgrid_size,BLOCK_SIZE>>>( gpu_count,k,gpu_centroid_y,gpu_centroid_z,gpu_centroid_x,gpu_sum_y,gpu_sum_z,gpu_sum_x);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaMemset(gpu_cluster_size, 0, k * sizeof(int)));
    CUDA_CHECK(cudaMemset(gpu_histogram, 0, k * 256 * sizeof(int)));

    kmeans_build_histogram<<<grid_size,BLOCK_SIZE>>>(n,k,gpu_assignments,gpu_I,gpu_histogram,gpu_cluster_size);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    make_equal_km_histogram<<<grid_size,BLOCK_SIZE>>>( n,gpu_I,gpu_assignments,gpu_cluster_size,gpu_histogram,gpu_output);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int *host_output = nullptr;
    CUDA_CHECK(cudaMallocHost(&host_output, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(host_output, gpu_output, n * sizeof(int), cudaMemcpyDeviceToHost));

    write_output_file("kmeans.txt", host_x, host_y, host_z, host_output, n);
    CUDA_CHECK(cudaFree(gpu_sum_y));CUDA_CHECK(cudaFree(gpu_centroid_x));     CUDA_CHECK(cudaFree(gpu_assignments)); CUDA_CHECK(cudaFree(gpu_count)); CUDA_CHECK(cudaFreeHost(host_output));CUDA_CHECK(cudaFree(gpu_sum_z));CUDA_CHECK(cudaFree(gpu_sum_x)); CUDA_CHECK(cudaFree(gpu_changed_flag)); CUDA_CHECK(cudaFree(gpu_centroid_y));CUDA_CHECK(cudaFree(gpu_cluster_size));CUDA_CHECK(cudaFree(gpu_centroid_z));CUDA_CHECK(cudaFree(gpu_histogram)); CUDA_CHECK(cudaFree(gpu_output));
    
}

int main(int argc, char *argv[]) {
    if (argc < 3) 
    {
        fprintf(stderr, "Usage: %s input.txt [knn|approx_knn|kmeans]\n", argv[0]);
        return 1;
    }

    const char *mode = argv[2];
    FILE *fin = fopen(argv[1], "r");
    if (!fin) 
    {
        fprintf(stderr, "Cannot open %s\n", argv[1]);
        return 1;
    }

    int n, k, T;
    if (fscanf(fin, "%d %d %d", &n, &k, &T) != 3) 
    {
        fprintf(stderr, "Bad input format\n");
        fclose(fin);
        return 1;
    }

    int *host_x = (int *)malloc(n * sizeof(int));
    if (!host_x)
     {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    int *host_y = (int *)malloc(n * sizeof(int));
    if (!host_y)
     {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    int *host_z = (int *)malloc(n * sizeof(int));
    if (!host_z)
     {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    int *hI = (int *)malloc(n * sizeof(int));
    if (!hI) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    rep(i,0,n) fscanf(fin, "%d %d %d %d", &host_x[i], &host_y[i], &host_z[i], &hI[i]); 
    fclose(fin); int *gpu_x = nullptr; 

    CUDA_CHECK(cudaMalloc(&gpu_x, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(gpu_x, host_x, n * sizeof(int), cudaMemcpyHostToDevice));

    int  *gpu_y = nullptr;
    CUDA_CHECK(cudaMalloc(&gpu_y, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(gpu_y, host_y, n * sizeof(int), cudaMemcpyHostToDevice));

    int  *gpu_z = nullptr;
    CUDA_CHECK(cudaMalloc(&gpu_z, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(gpu_z, host_z, n * sizeof(int), cudaMemcpyHostToDevice));

    int  *dI = nullptr;
    CUDA_CHECK(cudaMalloc(&dI, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dI, hI, n * sizeof(int), cudaMemcpyHostToDevice));

    if (strcmp(mode, "knn") == 0) {
        run_knn(host_x, host_y, host_z, hI, gpu_x, gpu_y, gpu_z, dI, n, k);
    }
    else if (strcmp(mode, "approx_knn") == 0) {
        run_approx_knn(host_x, host_y, host_z, hI, gpu_x, gpu_y, gpu_z, dI, n, k);
    }
    else if (strcmp(mode, "kmeans") == 0) {
        run_kmeans(host_x, host_y, host_z, hI, gpu_x, gpu_y, gpu_z, dI, n, k, T);
    }
    else {
        fprintf(stderr, "Invalid mode: %s\n", mode);
        free(host_x); free(host_y); free(host_z); free(hI);
        CUDA_CHECK(cudaFree(gpu_x));
        CUDA_CHECK(cudaFree(gpu_y));
        CUDA_CHECK(cudaFree(gpu_z));
        CUDA_CHECK(cudaFree(dI));
        return 1;
    }

    free(host_x);
    CUDA_CHECK(cudaFree(gpu_y));

    free(host_y);
    CUDA_CHECK(cudaFree(gpu_z));

    free(host_z);
    CUDA_CHECK(cudaFree(gpu_x));

    free(hI);
    CUDA_CHECK(cudaFree(dI));
    
    return 0;
}