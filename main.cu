#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>




// Simple transformation kernel
__global__ void transformKernel(float* output, cudaTextureObject_t coolTexObj, cudaTextureObject_t heatTexObj, int nx, int ny, float log_n)
{
  // Calculate normalized texture coordinates
  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  int yid = blockIdx.y * blockDim.y + threadIdx.y;
  float t = float(xid)/nx;
  float n = (log_n + 6.0)/12.1;
  if (xid < nx && yid < ny) {
    // Read from texture and write to global memory
    output[yid*nx + xid] = tex2D<float>(coolTexObj, t, n);
    output[nx*ny + yid*nx + xid] = tex2D<float>(heatTexObj, t, n);
    //printf("%3d %d %f %f %f %f\n", xid, yid, n, t, output[yid*nx + xid], output[nx*ny + yid*nx + xid]);
  }
}

void Load_Cooling_Tables(float* cooling_table, float* heating_table);

double get_time(void);


// Host code
int main()
{
  float *cooling_table;
  float *heating_table;
  float *h_output;
  const int nx = 81;
  const int ny = 121;
  int nx_out = 1000;
  int ny_out = 1; 
  float log_n = -3.0;
  double start_t, stop_t;


  // allocate arrays to be copied to textures
  cooling_table = (float *) malloc(nx*ny*sizeof(float));
  heating_table = (float *) malloc(nx*ny*sizeof(float));

  // Load cooling table into the array
  Load_Cooling_Tables(cooling_table, heating_table);

  // allocate output array on host
  h_output = (float *) malloc(2*nx_out*ny_out*sizeof(float));
  for (int i=0; i<nx_out; i++) {
    for (int j=0; j<ny_out; j++) {
      h_output[i+nx_out*j] = 0.0;
    }
  }

  // Allocate array to store result of transformations in device memory
  float* output;
  cudaMalloc(&output, 2*nx_out*ny_out*sizeof(float));

  // set info for cuda kernels
  dim3 dimBlock(16, 16);
  dim3 dimGrid((nx_out + dimBlock.x - 1) / dimBlock.x, (ny_out + dimBlock.y - 1) / dimBlock.y, 1);


  // Allocate CUDA array in device memory
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaArray* cuCoolArray;
  cudaArray* cuHeatArray;
  cudaMallocArray(&cuCoolArray, &channelDesc, nx, ny);
  cudaMallocArray(&cuHeatArray, &channelDesc, nx, ny);
  // Copy to device memory the cooling and heating arrays
  // in host memory
  cudaMemcpyToArray(cuCoolArray, 0, 0, cooling_table, nx*ny*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToArray(cuHeatArray, 0, 0, heating_table, nx*ny*sizeof(float), cudaMemcpyHostToDevice);

  // Specify textures
  struct cudaResourceDesc coolResDesc;
  memset(&coolResDesc, 0, sizeof(coolResDesc));
  coolResDesc.resType = cudaResourceTypeArray;
  coolResDesc.res.array.array = cuCoolArray;
  struct cudaResourceDesc heatResDesc;
  memset(&heatResDesc, 0, sizeof(heatResDesc));
  heatResDesc.resType = cudaResourceTypeArray;
  heatResDesc.res.array.array = cuHeatArray;  

  // Specify texture object parameters (same for both tables)
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp; // out-of-bounds fetches return border values
  texDesc.addressMode[1] = cudaAddressModeClamp; // out-of-bounds fetches return border values
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 1;

  // Create texture objects
  cudaTextureObject_t coolTexObj = 0;
  cudaCreateTextureObject(&coolTexObj, &coolResDesc, &texDesc, NULL);
  cudaTextureObject_t heatTexObj = 0;
  cudaCreateTextureObject(&heatTexObj, &heatResDesc, &texDesc, NULL);

  // Invoke kernel
  start_t = get_time();  
  transformKernel<<<dimGrid, dimBlock>>>(output, coolTexObj, heatTexObj, nx_out, ny_out, log_n);
  stop_t = get_time();
  //printf("%f ms\n", (stop_t-start_t)*1000);  
  cudaDeviceSynchronize();

  // Copy the results back to the host
  cudaMemcpy(h_output, output, 2*nx_out*ny_out*sizeof(float), cudaMemcpyDeviceToHost);


  for (int j=0; j<ny_out; j++) {
    for (int i=0; i<nx_out; i++) {
      printf("%6.3f %6.3f\n", h_output[j*nx_out + i], h_output[nx_out*ny_out + j*nx_out + i]);
    }
  }  


  // Destroy texture object
  cudaDestroyTextureObject(coolTexObj);
  cudaDestroyTextureObject(heatTexObj);
  // Free device memory
  cudaFreeArray(cuCoolArray);
  cudaFreeArray(cuHeatArray);
  cudaFree(output);
  // Free host memory
  free(cooling_table);
  free(heating_table);
  free(h_output);

  return 0;

}



void Load_Cooling_Tables(float* cooling_table, float* heating_table)
{
  double *n_arr;
  double *T_arr;
  double *L_arr;
  double *H_arr;

  int i;
  int nx = 121;
  int ny = 81;

  FILE *infile;
  char buffer[0x1000];
  char * pch;

  // allocate arrays for temperature data
  n_arr = (double *) malloc(nx*ny*sizeof(double));
  T_arr = (double *) malloc(nx*ny*sizeof(double));
  L_arr = (double *) malloc(nx*ny*sizeof(double));
  H_arr = (double *) malloc(nx*ny*sizeof(double));

  // Read in cloudy cooling/heating curve (function of density and temperature)
  i=0;
  infile = fopen("./cloudy_coolingcurve.txt", "r");
  if (infile == NULL) {
    printf("Unable to open Cloudy file.\n");
    exit(1);
  }
  while (fgets(buffer, sizeof(buffer), infile) != NULL)
  {
    if (buffer[0] == '#') {
      continue;
    }
    else {
      pch = strtok(buffer, "\t");
      n_arr[i] = atof(pch);
      while (pch != NULL)
      {
        pch = strtok(NULL, "\t");
        if (pch != NULL)
          T_arr[i] = atof(pch);
        pch = strtok(NULL, "\t");
        if (pch != NULL)
          L_arr[i] = atof(pch);
        pch = strtok(NULL, "\t");
        if (pch != NULL)
          H_arr[i] = atof(pch);
      }
      i++;
    }
  }
  fclose(infile);

  // copy data from cooling array into the table
  for (i=0; i<nx*ny; i++)
  {
    cooling_table[i] = float(L_arr[i]);
    heating_table[i] = float(H_arr[i]);
  }

  // Free arrays used to read in table data
  free(n_arr);
  free(T_arr);
  free(L_arr);
  free(H_arr);
}


double get_time(void)
{ 
  struct timeval timer;
  gettimeofday(&timer,NULL);
  return timer.tv_sec + 1.0e-6*timer.tv_usec;
}
