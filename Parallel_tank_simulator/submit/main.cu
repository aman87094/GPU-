#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>

using namespace std;

//*******************************************

// Write down the kernels here
#define MAX_TANKS 1000

__device__ int calculateDistance(int x1, int y1, int x2, int y2) {
    return (x1 - x2)*(x1 - x2)+(y1 - y2)*(y1 - y2);
}

__global__ void simulateGame(int M, int N, int T, int* tankX, int* tankY, int* tankHP, int* tankScore,volatile int* d_cnt,int round) {
    int tankID = blockIdx.x * blockDim.x + threadIdx.x;
    if (tankID >T || tankHP[tankID] <= 0) return;


    int targetID = (tankID + round) % T;
    if (targetID == tankID) return; 
    
    int dx = tankX[targetID] - tankX[tankID];
    int dy = tankY[targetID] - tankY[tankID];
    
    // bool blocked = false;
    int ind=-1;
    int dis=1e9;
    for (int i = 0; i < T; ++i) {
        // if (i == tankID || i == targetID) continue;
        if(i==tankID) continue;
        int dxi = tankX[i] - tankX[tankID];
        int dyi = tankY[i] - tankY[tankID];
        // && calculateDistance(tankX[tankID], tankY[tankID], tankX[i], tankY[i]) < calculateDistance(tankX[tankID], tankY[tankID], tankX[targetID], tankY[targetID])
        if (dx * dyi == dy * dxi && ((tankX[targetID]-tankX[tankID])*(tankX[i]-tankX[tankID])>=0 &&(tankY[targetID]-tankY[tankID])*(tankY[i]-tankY[tankID])>=0)) {
            int curdst = calculateDistance(tankX[tankID], tankY[tankID], tankX[i], tankY[i]);
            if(curdist<dis && tankScore[i]>0){
                ind=i;
                dis=curdist;
            }
        }
    }
    __syncthreads();
    if (ind!=-1) {
        // tankScore[tankID]++;
        atomicAdd(&tankScore[tankID], 1);
        // tankHP[targetID]--;
        atomicAdd(&d_cnt,1);
        atomicAdd(&tankHp[ind], -1);
    }
    
}

//***********************************************


int main(int argc,char **argv)
{
    // Variable declarations
    int M,N,T,H,*xcoord,*ycoord,*score;
    

    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &M );
    fscanf( inputfilepointer, "%d", &N );
    fscanf( inputfilepointer, "%d", &T ); // T is number of Tanks
    fscanf( inputfilepointer, "%d", &H ); // H is the starting Health point of each Tank
	
    // Allocate memory on CPU
    xcoord=(int*)malloc(T * sizeof (int));  // X coordinate of each tank
    ycoord=(int*)malloc(T * sizeof (int));  // Y coordinate of each tank
    score=(int*)malloc(T * sizeof (int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for(int i=0;i<T;i++)
    {
      fscanf( inputfilepointer, "%d", &xcoord[i] );
      fscanf( inputfilepointer, "%d", &ycoord[i] );
    }
		

    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************

    int* tankHp=(int*)malloc(T * sizeof (int));

    for(int i=0;i<T;i++){
        tankHp[i]=H;
        score[i]=0;
    }

    int *d_tankX, *d_tankY, *d_tankHP, *d_tankScore;
    cudaMalloc(&d_tankX,T*sizeof(int));
    cudaMalloc(&d_tankY,T*sizeof(int));
    cudaMalloc(&d_tankHP,T*sizeof(int));
    cudaMalloc(&td_tankScore,T*sizeof(int));

    cudaMemcpy(d_tankX, tankX, T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tankY, tankY, T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tankHP, tankHP, T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tankScore, tankScore, T * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (T + blockSize - 1) / blockSize;

    int cnt=0;
    int *d_cnt;
    cudaMalloc(&d_cnt,sizeof(int));
    cudaMemcpy(d_cnt,cnt,sizeof(int),cudaMemcpyHostToDevice);
    int round=1;

    do{
        cudaMemset(d_cnt,0,sizeof(int));
        cudaDeviceSynchronize();
        simulateGame<<<1, T>>>(M, N, T, d_tankX, d_tankY, d_tankHP, d_tankScore,d_cnt,round);
        cudaMemcpy(cnt,d_cnt,sizeof(int),cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        round++;

    }while(cnt>=2);

    

    cudaMemcpy(tankScore, d_tankScore, T * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Tank scores:\n");
    for (int i = 0; i < T; ++i) {
        printf("Tank %d: %d\n", i, tankScore[i]);
    }

    cudaFree(d_tankX);
    cudaFree(d_tankY);
    cudaFree(d_tankHP);
    cudaFree(d_tankScore);


    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end  = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end-start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char *outputfilename = argv[2];
    char *exectimefilename = argv[3]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    for(int i=0;i<T;i++)
    {
        fprintf( outputfilepointer, "%d\n", score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename,"w");
    fprintf(outputfilepointer,"%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;
}