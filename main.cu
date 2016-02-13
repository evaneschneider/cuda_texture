#include<stdio.h>



// Simple transformation kernel
__global__ void transformKernel(float* output, cudaTextureObject_t texObj, int nx, int ny)
{
  // Calculate normalized texture coordinates
  int xid = blockIdx.x * blockDim.x + threadIdx.x;
  int yid = blockIdx.y * blockDim.y + threadIdx.y;
  float x = (xid+0.5f)/nx;
  float y = (yid+0.5f)/ny;
  if (xid < nx && yid < ny) {
    //printf("%f %f\n", x, y);
    // Read from texture and write to global memory
    output[yid * nx + xid] = tex2D<float>(texObj, x, y);
  }
}


// Host code
int main()
{
  float *h_data;
  float *h_output;
  const int nx = 5;
  const int ny = 5;
  int nx_out = 10;
  int ny_out = 10; 


  // allocate table on host
  h_data = (float *) malloc(nx*ny*sizeof(float));
  for (int j=0; j<ny; j++) {
    for (int i=0; i<nx; i++) {
      h_data[i+nx*j] = (i*nx+j*ny)/nx;
      printf("%5.2f ", h_data[i+nx*j]);
    }
    printf("\n");
  }

  // allocate output array on host
  h_output = (float *) malloc(nx_out*ny_out*sizeof(float));
  for (int i=0; i<nx_out; i++) {
    for (int j=0; j<ny_out; j++) {
      h_output[i+nx_out*j] = 0.0;
    }
  }
 
  // set info for cuda kernels
  dim3 dimBlock(16, 16);
  dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y, 1);


  // Allocate CUDA array in device memory
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaArray* cuArray;
  cudaMallocArray(&cuArray, &channelDesc, nx, ny);
  // Copy to device memory some data located at address h_data
  // in host memory
  cudaMemcpyToArray(cuArray, 0, 0, h_data, nx*ny*sizeof(float), cudaMemcpyHostToDevice);

  // Specify texture
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;

  // Specify texture object parameters
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp; // out-of-bounds fetches return border values
  texDesc.addressMode[1] = cudaAddressModeClamp; // out-of-bounds fetches return border values
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 1;

  // Create texture object
  cudaTextureObject_t texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

  // Allocate result of transformation in device memory
  float* output;
  cudaMalloc(&output, nx_out * ny_out * sizeof(float));

  // Invoke kernel
  printf("About to perform texture read...\n");
  transformKernel<<<dimGrid, dimBlock>>>(output, texObj, nx_out, ny_out);
  cudaDeviceSynchronize();
  printf("Texture read complete.\n");

  cudaMemcpy(h_output, output, nx_out*ny_out*sizeof(float), cudaMemcpyDeviceToHost);

  for (int j=0; j<ny_out; j++) {
    for (int i=0; i<nx_out; i++) {
      printf("%5.2f ", h_output[i+nx_out*j]);
    }
    printf("\n");
  }  

  // Destroy texture object
  cudaDestroyTextureObject(texObj);
  // Free device memory
  cudaFreeArray(cuArray);
  cudaFree(output);
  // Free host memory
  free(h_data);
  free(h_output);

  return 0;

}
