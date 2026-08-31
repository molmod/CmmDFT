.. _sec:Solver:

Numerical solving algorithms
============================

This section describes the numerical methods used to solve the equilibrium density-functional equations, including iteration schemes, convergence criteria, mixing, and stabilization methods.

Several algorithms were implemented and tested to solve the DFT equations, these include:

-  Picard line search

-  Fast inertial relaxation engine (FIRE)

-  Anderson acceleration

-  Quasi-Newton (QN) methods: L-BFGS, BROYDEN

-  Conjugate gradient (CG) descent

The grand potential functional employed in CmmDFT approximates the exact functional, and the combination of this approximation and numerical discretization can introduce spurious minima or cause divergent behavior during iterative solution. It is therefore crucial to develop robust and efficient solution methods capable of detecting and avoiding convergence to these spurious minima. Standard minimization routines, such as the L-BFGS ond the conjugate gradient solvers available in SciPy, were found to be inadequate for this purpose, as they frequently converged to non-physical solutions or diverged entirely.

Each method (except FIRE) was used in a hybrid algorithm with Picard iterations to keep the initial steps stable, :cite:`edelmann2016nummerical, sang2021gas` because Picard iterations are reliable and help to avoid non-physical steps when the density is still far from equilibrium. The switch from Picard iterations to the primary solver is triggered by a criterion based on the residual norm; specifically, the switch is executed once the residual norm has decreased to half its value at the first iteration.

Picard iterations were also used as a fallback whenever the primary solver took a bad step. A step is classified as bad if any of the following conditions are met: (i) non-finite values appear in the loading or density update, (ii) the residual norm increases by more than a factor :math:`2` relative to the previous iteration, or (iii) the grand potential increases by more than :math:`1.5` in a single step. This fallback mechanism substantially increased the robustness of the algorithms with minimal loss of efficiency. To guard against rare numerical instabilities that cause complete failure of a calculation, a failsafe mechanism is implemented: the simulation is automatically restarted with the mixing parameter reduced by a factor of two. The definition of the mixing parameter is given for each method in the subsections below. Although this measure is only rarely triggered, it effectively prevents divergent calculations from failing outright.

.. _sec:Picard:

Line search Picard iterations
-----------------------------

One approach to solving Equation \ `[eq:euler_lagrange_equation] <#eq:euler_lagrange_equation>`__ is to iterate from an initial density estimate, repeatedly applying :math:`G` to generate successive updates. To improve numerical stability, the newly generated density is mixed with the previous iterate:

.. math:: \rho^{(k+1)}(\mathbf{r}) = \lambda^{(k)} G[\mathbf{r};\rho^{(k)}] + (1-\lambda^{(k)})\rho^{(k)}(\mathbf{r}),

where :math:`\lambda^{(k)} \in (0,1]` is the mixing parameter at iteration :math:`k`, controlling the trade-off between stability and convergence speed. A large :math:`\lambda` can induce numerical instabilities and unphysical density profiles, while a small :math:`\lambda` leads to slow convergence. A first safeguard is to ensure that :math:`\lambda` never exceeds the value at which the local packing fraction in the FMT functional, :math:`n_3(\vec{r})`, would surpass unity, which is physically impossible. The mixing parameter is therefore restricted to the interval for which :math:`n_3(\vec{r}) < 1` is guaranteed throughout the domain. :cite:`roth2010fundamental`

For more efficient convergence, :math:`\lambda` is selected at each iteration via a line search. The default method, proposed by Roth, :cite:`roth2010fundamental` estimates the optimal :math:`\lambda` by evaluating the grand potential :math:`\Omega` at a small number of candidate values, fitting the result to a quadratic function, and taking the minimizing :math:`\lambda` as the update step. This approach is computationally inexpensive and performs well in the early stages of iteration. However, as the density approaches equilibrium, the variation in :math:`\Omega` across candidate values becomes small, degrading the quadratic fit and occasionally yielding non-positive :math:`\lambda` estimates.

When this occurs, the algorithm switches to one of two fallback strategies based on the spread of :math:`\Omega` values sampled during the quadratic step. If the difference between the maximum and minimum sampled values exceeds 1 kJ/mol, a sequential least squares quadratic programming (SLSQP) line search is invoked, which minimizes :math:`\Omega` directly as a function of :math:`\lambda` at higher computational cost. Otherwise, a fixed value of :math:`\lambda = 0.1` is used, which was found through numerical testing to balance stability and convergence speed in this near-equilibrium regime.

.. _sec:FIRE:

Fast inertial relaxation engine
-------------------------------

The Fast Inertial Relaxation Engine (FIRE) algorithm :cite:`bitzek2006structural` is a minimization method inspired by molecular dynamics, which introduces a fictitious velocity field :math:`V^{(k)}` that is updated using the functional derivative of the grand potential as a driving force. Rather than computing an explicit search direction from gradient or curvature information, FIRE accelerates convergence by building up inertia along favorable directions and dissipating it when the trajectory moves away from the minimum. It was applied to cDFT by Soares et al. in pydftlj, :cite:`soares2023classical` it is adapted in CmmDFT to compare its performance against other numerical algorithms.

The density is updated by updating :math:`\ln\rho` with a velocity-Verlet integrator, where the force is defined as :math:`F^{(k)} = -\beta\frac{\delta\Omega}{\delta\rho}[\rho^{(k)}]`. Working with the logarithm of the density naturally enforces positivity of :math:`\rho` throughout the process. At each iteration :math:`P^{(k)} = F^{(k)} \cdot V^{(k)}` is monitored to assess whether the trajectory is moving toward or away from the minimum. When :math:`P^{(k)} > 0` the system is moving downhill, the timestep is increased and :math:`\lambda^{(k)}` is slowly reduced to allow faster convergence:

.. math::

   \begin{aligned}
       \Delta t^{(k+1)} &= \min\left(\Delta t^{(k)} f_{\text{inc}},\ \Delta t_{\text{max}}\right)\\
       \lambda^{(k+1)} &= \max\left(\lambda^{(k)} f_\lambda,\ \lambda_{\text{min}}\right),\end{aligned}

where :math:`f_{\text{inc}} = 1.1` and :math:`f_\lambda = 0.99`. When :math:`P^{(k)} \leq 0` the trajectory has overshot. To avoid uphill motion, the previous half-step is rolled back, the velocity is reset to zero, the timestep is reduced and :math:`\lambda^{(k)}` is reset to its initial value :math:`\lambda_0`:

.. math::

   \begin{aligned}
       \ln\rho^{(k)} &\leftarrow \ln\rho^{(k)} - V^{(k)} \cdot \frac{\Delta t^{(k)}}{2}\\
       V^{(k)} &\leftarrow 0\\
       \Delta t^{(k+1)} &= \max\left(\Delta t^{(k)} f_{\text{dec}},\ \Delta t_{\text{min}}\right),\end{aligned}

where :math:`f_{\text{dec}} = 0.5`. After the rollback the force is recomputed at the restored position before continuing. Following this, the positions and velocities are update using a dynamical integrator. Here a velocity-Verlet integrator is implemented, following Guénolé et al. :cite:`guenole2020assessment` The update proceeds as follows:

.. math::

   \begin{aligned}
       V^{(k+\frac{1}{2})} &= V^{(k)} + F^{(k)} \cdot \frac{\Delta t^{(k)}}{2}\\
       \hat{V}^{(k+\frac{1}{2})} &= \left(1 - \lambda^k\right)V^{(k+\frac{1}{2})} + 
                   \lambda^{(k)} F^{(k)} \frac{\|V^{(k+\frac{1}{2})}\|}{\|F^{(k)}\|}\\
       \ln\rho^{(k+1)} &= \ln\rho^{(k)} + \Delta t^{(k)} \cdot \hat{V}^{(k+\frac{1}{2})}\\
       V^{(k+1)} &= \tilde{V}^{(k+\frac{1}{2})} + F^{(k+1)} \cdot \frac{\Delta t^{(k)}}{2},\end{aligned}

A mixing step is introduced with the adaptive mixing factor :math:`\lambda` where :math:`F^{(k+1)} = -\beta\frac{\delta\Omega}{\delta\rho}[\rho^{(k+1)}]` is the force evaluated at the new position. In CmmDFT the ABC-FIRE variant :cite:`restrepo2023abc` is implemented, which introduces an additional acceleration factor applied to the mixed velocity before the position update:

.. math::

   \begin{aligned}
       \hat{V}^{(k+\frac{1}{2})} &= \frac{\hat{V}^{(k+\frac{1}{2})}}{1-(1-\lambda^{(k)})^{N_{\text{pos}}}},\end{aligned}

where :math:`N_{\text{pos}}` is the number of consecutive steps for which :math:`P^{(k)} > 0`, correcting for the bias introduced by the damping toward the force direction. This factor mainly speeds up the initial steps of the algorithm.

.. _sec:Anderson:

Anderson acceleration
---------------------

Anderson acceleration :cite:`anderson1965iterative` is a fixed point iteration that computes a new density :math:`\rho^{(k+1)}` as a linear combination of :math:`m^{(k)}` previous densities (:math:`\rho^{(k)}`, :math:`\rho^{(k-1)}`, ...) as well as their :math:`G`-evaluation (:math:`G[\rho^{(k)}]`, :math:`G[\rho^{(k-1)}]`, ...)

.. math::

   \begin{aligned}
       \rho^{(k+1)}(\vec{r}) &= (1-\lambda^{(k)})\sum_{i=0}^{m^{(k)}}\alpha^{(k)}_i \rho^{(k-m^{(k)}+i)}(\vec{r}) + \lambda^{(k)}\sum_{i=0}^{m^{(k)}}\alpha^{(k)}_i G[\vec{r},\rho^{(k-m^k+i)}]\end{aligned}

Where :math:`m^{(k)}` is :math:`\min(k,m)`, with :math:`m` the maximum number of previous densities taken into account, :math:`\lambda^{(k)}` is a mixing parameter and :math:`\alpha_i` are the coefficients determined by minimizing the norm of the residuals:

.. math::

   \begin{aligned}
       \vec{\alpha}^{(k)} = argmin_{\vec{\alpha}}&\left|\sum_{i=0}^{m^{(k)}}\alpha_i R[\vec{r};\rho^{(k-m^{(k)}+i)}]\right|_2,\end{aligned}

subject to the following constraints:

.. math::

   \begin{aligned}
        \left(
           \begin{array}{l}
               \sum_{i=0}^{m^{(k)}}\alpha_i = 1 \\
               \alpha_i \geq 0
           \end{array}
         \right).\end{aligned}

This least squares problem was solved using the SLSQP solver implemented in Scipy, as this method showed great performance and allows for application of constraints on the mixing parameters.

In this study the damping parameter, :math:`\lambda^{(k)}`, was variable from step to step, with an adaptive damping scheme based on the evolution of the residual norm: if the residual increased in the previous step, the damping parameters was decreased with factor :math:`0.6`, otherwise it was increased by factor :math:`1.2`. The damping parameter was limited to a maximum of 0.8 to ensure stability and a minimum of 0.01 to avoid stalling. The number of previous densities :math:`m`, should not be set too high, as the least squares problem can become ill-conditioned and the solver will require more memory, however limiting this number too much can cause the algorithm to lose efficiency, as it can not optimally determine the search direction. In this study a value of 8 was chosen for :math:`m`.

This method is relatively easy to implement and does not have a large number of parameters to tune. Additionally it benefits from the fact that no extra computations of the excess functional or its derivative are required to obtain optimal mixing coefficients, making a single step more efficient. In general it proved to be a robust and efficient numerical solver.

.. _sec:QN:

Quasi Newton methods
--------------------

Quasi-Newton methods are algorithms used to find roots or local extrema of functions by iteratively updating an approximation of the inverse Hessian or Jacobian, without requiring the exact second-order derivatives needed by Newton’s method. :cite:`dennis1977quasi` In CmmDFT, the equilibrium density is found by directly minimizing :math:`\Omega[\rho]`, using the functional derivatives :math:`g^{(k)} = \frac{\delta\Omega}{\delta\rho}[\rho^{(k)}]` to form an estimate :math:`B^{(k)}` of the inverse Hessian of :math:`\Omega[\rho]`. The density is updated as:

.. math::

   \begin{aligned}
       \rho^{(k+1)} &= \rho^{(k)} + \lambda^{(k)} d^{(k)},\\
       d^{(k)} &= -B^{(k)} g^{(k)},\end{aligned}

where :math:`d^{(k)}` is the search direction and :math:`\lambda^{(k)}` is a step size determined by a backtracking line search until the Armijo sufficient decrease condition is satisfied. :cite:`armijo1966minimization` To ensure :math:`\rho^{(k+1)}(\mathbf{r}) \geq 0` throughout the domain, any components of the search direction :math:`d^{(k)}` that would drive the density negative are set to zero prior to the line search. At each iteration the density displacement and the difference in the gradient are stored to update the inverse Hessian approximation:

.. math::

   \begin{aligned}
       s^{(k)} &= \rho^{(k+1)} - \rho^{(k)},\\
       y^{(k)} &= g^{(k+1)} - g^{(k)},\end{aligned}

Two update schemes are implemented in CmmDFT: the limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) :cite:`nocedal1980updating, mairhofer2017numerical` and the limited-memory Broyden (L-Broyden) :cite:`broyden1965class, edelmann2016nummerical` methods, with update equations:

.. math::

   \begin{aligned}
       B_\mathrm{BFGS}^{(k+1)} &= \left(\mathbf{I} - \frac{s^{(k)}{y^{(k)}}^T}{{y^{(k)}}^T s^{(k)}}\right)B^{(k)}
                         \left(\mathbf{I} - \frac{y^{(k)}{s^{(k)}}^T}{{y^{(k)}}^T s^{(k)}}\right)
                         + \frac{s^{(k)}{s^{(k)}}^T}{{y^{(k)}}^T s^{(k)}},\\
       B_\mathrm{Broyden}^{(k+1)} &= B^{(k)} + \frac{\left(s^{(k)} - B^{(k)}y^{(k)}\right){s^{(k)}}^T B^{(k)}}{{s^{(k)}}^T B^{(k)} y^{(k)}}.\end{aligned}

In the limited-memory variants, only the last :math:`m^{(k)} = \min(m, k)` iteration pairs :math:`\left(s^{(k)}, y^{(k)}\right)` are stored and applied implicitly via the two-loop recursion, avoiding explicit construction of :math:`B^{(k)}`. In CmmDFT, :math:`m = 10` is used. To improve robustness in the early iterations or when the quasi-Newton step fails to satisfy the Armijo condition, the solver falls back to a Picard step, as discussed earlier.

.. _sec:CG:

Conjugate gradient descent
--------------------------

Nonlinear conjugate gradient methods are large-scale optimization algorithms for finding local minima of smooth functionals using only first-order derivative information. Unlike quasi-Newton methods, conjugate gradient schemes do not build or store an explicit approximation of the inverse Hessian. Instead, they construct a sequence of mutually conjugate search directions by combining the current gradient with information from previous iterations, leading to low memory requirements and good scalability for high-dimensional problems. In CmmDFT, the CG-Descent method of Hager and Zhang is implemented. :cite:`hager2005new` The density is updated as:

.. math::

   \begin{aligned}
   \rho^{(k+1)} &= \rho^{(k)} + \lambda^{(k)} d^{(k)},\end{aligned}

where :math:`d^{(k)}` is the conjugate search direction and :math:`\lambda^{(k)} > 0` is the step size. The search direction is initialized as the steepest descent direction,

.. math::

   \begin{aligned}
   d^{(0)} &= -g^{(0)} = -\frac{\delta\Omega}{\delta\rho}[\rho^{(0)}],\end{aligned}

and updated recursively for subsequent iterations as

.. math::

   \begin{aligned}
   d^{(k)} &= -g^{(k)} + \beta^{(k)} d^{(k-1)}.\end{aligned}

The CG-Descent method uses a specific choice of the conjugacy parameter :math:`\beta^{(k)}` designed to guarantee descent independently of the accuracy of the line search. Using the gradient difference defined earlier: :math:`y^{(k)} = g^{(k)} - g^{(k-1)}`, the update parameter is given by

.. math::

   \begin{aligned}
       \beta^{(k)} = \frac{1}{{d^{(k-1)}}^T y^{(k)}}\left(y^{(k)} - 2 d^{(k-1)} \frac{|y^{(k)}|^2}{{d^{(k-1)}}^T y^{(k)}}\right)^T g^{(k)}.\end{aligned}

To ensure numerical robustness and preserve the descent property, :math:`\beta^{(k)}` is truncated if necessary such that :math:`({g^{(k)}})^T d^{(k)} < 0`. As in the quasi-Newton implementation, any components of :math:`d^{(k)}` that would drive the density negative are set to zero prior to the line search, ensuring :math:`\rho^{(k+1)}(\vec{r}) \geq 0` throughout the domain.

.. _sec:Conv:

Convergence criteria
--------------------

A challenge in the implementation of CmmDFT is finding the right convergence criterion. Currently three criteria are implemented in CmmDFT, relative integrated unsigned error (RIUE), norm of the residual (RES) and the norm of the derivative of the grand potential (DER):

.. math::

   \begin{aligned}
   {3}
       &\text{RIUE:} \quad 
           &&\frac{\int|\rho^{(k)}-\rho^{(k-1)}|}{\int \rho^{(k)}} 
           &&< \epsilon \\
       &\text{RES:} \quad 
           &&\frac{\left|G(\rho^{(k)})-\rho^{(k)}\right|_2}{\sqrt{N_{grid}N_{ads}}} 
           &&< \epsilon \\
       &\text{DER:} \quad 
           &&\left\|\frac{\beta\rho^{(k)}\frac{\delta\Omega}{\delta\rho}[\rho^{(k)}]}{a_{tol} + r_{tol}\rho^{(k)}}\right\|_2\frac{1}{\sqrt{N_{grid}}} 
           &&< 1\end{aligned}

Each criterion carries distinct advantages and limitations. RIUE is straightforward to implement and interpret, requires no additional computation, but is susceptible to numerical noise. RES is widely used and generally robust, as it is insensitive to the previous iterate and less affected by noise. However, it can be misleading at low pressures, where it where it often leads to premature termination, leading to significant errors in the converged density. DER is theoretically appealing, as it directly reflects the first-order optimality condition for a local minimum, but in practice it was found to be unreliable at low guest densities, where the gradient of the grand potential can exhibit large spurious fluctuations near framework atoms.

The choice of convergence threshold is important, as it directly governs the trade-off between accuracy and computational cost. To formulate clear guidelines for present and future work, the impact of the convergence criterion and threshold on both adsorption isotherms and free energy profiles was systematically evaluated. The criteria RIUE, RES, and their combination (RES+RIUE) were tested with thresholds ranging from :math:`10^{-4}` to :math:`10^{-7}`.
