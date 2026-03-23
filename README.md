# Projected-Hypercube-Sampling-PHS

To address the existing limitations of reference vector generation method which is center biased and geometry dependent, proposed a Projected Hypercube Sampling (PHS) that arbitrarily samples in the uniform hypercube to densely cover the space and on their edges, project them onto the reference hyperplane, from this required number of reference vectors are selected using greedy equidistant approach,and finally these reference vectors are constructed normal to the reference hyperplane (0-1 direction) to intersect the Pareto surface.

In this approach, the number of initial samples on the hypercube and the number of points per edge are use defined.

Alpha parametrized convex Pareto front shapes
<img width="350" height="122" alt="image" src="https://github.com/user-attachments/assets/913391be-0413-41a8-af2d-00b0f03b1380" />

Das and Dennis systematic sampling
<img width="1829" height="1849" alt="SS sampling" src="https://github.com/user-attachments/assets/d490a24e-4b10-4cc4-824f-dda204f7ee44" />

Projected hypercube sampling (PHS)
<img width="1829" height="1858" alt="PHS sampling" src="https://github.com/user-attachments/assets/f5481449-7d0a-42c1-bd28-2cfc4e317cec" />


