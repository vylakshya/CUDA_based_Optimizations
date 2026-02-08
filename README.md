## Purpose
Implementing the GPU accelerated Algorithms and Optimizations through CUDA.
## 1. Tiled Matrix Multiplication
This implementation computes the product of two matrices $C = A \times B$ using a tiled approach.

Key Optimizations :

  -> Shared Memory Tiling: Instead of fetching elements directly from Global Memory for every calculation, threads load blocks (tiles) of data into shared memory. This reduces the global memory bandwidth bottleneck by a factor of the TILE_SIZE.
  
  -> Memory Coalescing: Data is loaded into shared memory using a pattern where threads in a warp access contiguous memory addresses, ensuring the hardware can combine multiple requests into a single transaction.
  
  -> Bank Conflict Avoidance: The TILE_SIZE is aligned with the GPU warp size (32), ensuring smooth access to shared memory banks.
  
  Usage:
  
  The kernel expects three pointers (d_A, d_B, d_C) and dimensions $M$, $N$, and $K$.
  
  Grid Layout:
  
  2D grid of 2D blocks.Complexity: $O(n^3)$ operations, but memory bound is significantly improved via tiling.
## 2. Numerical Integration
This implementation calculates the definite integral of a function using the Trapezoidal Rule.
Key Optimizations:

-> Grid-Stride Loops: Rather than assuming one thread per interval, threads loop over the range. This makes the code robust to any input size $N$ and keeps the GPU execution units busy without overhead.

-> Two-Stage Reduction: To avoid "Atomic Contention," threads first sum their values into registers, then perform a tree-based reduction in Shared Memory to find the block sum.

-> Atomic Aggregation: Only the final thread of each block performs an atomicAdd to global memory, minimizing the bottleneck on the final result variable.
