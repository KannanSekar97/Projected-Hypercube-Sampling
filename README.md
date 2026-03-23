# Projected-Hypercube-Sampling-PHS

To address the existing limitations of reference vector generation method which is center biased and geometry dependent, proposed a Projected Hypercube Sampling (PHS) that arbitrarily samples in the uniform hypercube to densely cover the space and on their edges, project them onto the reference hyperplane, from this required number of reference vectors are selected using greedy equidistant approach,and finally these reference vectors are constructed normal to the reference hyperplane (0-1 direction) to intersect the Pareto surface.

In this approach, the number of initial samples on the hypercube and the number of points per edge are use defined.


<table align="center">
  <tr>
    <td align="center">
      <img width="300" height="320" alt="Convex_alpha_1" src="https://github.com/user-attachments/assets/719d0971-0994-4a5b-b744-c3153a208d32" />
      <em>Convex PF $\alpha=1$</em>
    </td>
    <td align="center">
     <img width="300" height="320" alt="Convex_alpha_5" src="https://github.com/user-attachments/assets/f0efae3b-fe4a-41be-9f05-3adc681aa598" />
      <em>Convex PF $\alpha=5$</em>
    </td>
    <td align="center">
      <img width="300" height="320" alt="Convex_alpha_30" src="https://github.com/user-attachments/assets/66e74b74-2fac-44c2-bdf3-2095e93db607" />
      <em>Convex PF $\alpha=30$</em>
    </td>
  </tr>
</table>


<p align="center">
  <img width="320" height="340" alt="SS sampling" src="https://github.com/user-attachments/assets/d490a24e-4b10-4cc4-824f-dda204f7ee44" />
  <img width="320" height="340" alt="PHS sampling" src="https://github.com/user-attachments/assets/f5481449-7d0a-42c1-bd28-2cfc4e317cec" />
</p>
<p align="center">
  <em>Das and Dennis systematic sampling</em> &nbsp;&nbsp;&nbsp;
  <em>Projected hypercube sampling (PHS)</em> &nbsp;&nbsp;&nbsp;
</p>

