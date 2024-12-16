[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/560MyzRr)
# A5. Reduction and Histogram

Due Date: 11/13 at midnight

## Instructions

Implement the following approaches and compare their performance:

### Reduction

1. CPU Reduction
  - take various operators (sum, product, min, max) 
2. GPU Reduction
  - shared memory
  - multi-kernel, handle arrays > 2x shared mem size (> 2048)
3. GPU Reduction w/ less Thread Divergence

## Histogram

4. CPU Histogram
5. GPU Histogram - non-strided
6. GPU Histogram - strided

For both algorithms
you should collect timing data and analyze the results for increasing
array sizes. That is, at what point does each optimization pay off?
You only need to compare within an algorithms (1 v 2 v 3) and not across.

## Reflection Questions

2. What went well with this assignment?
3. What was difficult?
4. How would you approach differently?
5. Anything else you want me to know?

# Submission Checklist

[ ] Code (1-6)
[ ] Performance (table, graph)
[ ] Reflection
[ ] Peer feedback

# Debugging

If you encounter errors, I recommend two approahces:

1. Adding print statements both inside your kernel function
   as well as outside.
   This can include: `printf("%s\n", cudaGetErrorString(cudaGetLastError()));`
   to catch any cuda errors.

2. You can check to make sure that you code is not accessing unallocated memory
   by utilizing NVIDIA's memory sanitizer tool.
   You can run it ok `keroppi` using the following line.
   ```sh
   compute-sanitizer --tool memcheck ./your_cuda_executable_not_source
   ```

# Updates

To update this assignment as changes are made,
a new PR will be generated.
You can find the tab [here](../../pulls).
On that page you can merge the pull request to get the update instructions.
This may invovle rebasing or merging your contributions, reach out
if you need help with this.

# Peer Feedback

After the assignment is due,
you will be randomly assigned to review another student's submission.
This will be as a pull request (PR).
One is already opened by default ([PR #1](../../pull/1)) so you can leave your comments there.
You should look at the results, explanations, code, etc for
understandability, readability, style, etc and provide
any constructive or positive insights.
This peer feedback is due one week after the assignment closes.

# Extras

1. Calculate occupancy (see old calculators)
    - 2 v 3
    - 5 v 6
2. Implement histogram on GPU without atomics; is it faster?
