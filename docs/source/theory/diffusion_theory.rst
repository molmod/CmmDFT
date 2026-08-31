.. _sec:diffusion_method_fep:

Extracting free energy profiles of diffusion and diffusivities from cDFT
------------------------------------------------------------------------

In this section, we elaborate on how to extract a free energy profile of diffusion. To be more specific, we aim to construct profiles of the thermodynamic potential in the grand canonical ensemble, i.e. the grand potential, for a macrostate described in terms of a collective variable (CV) that quantifies the progress along the diffusion trajectory. However, as is often done in literature, we will denote such grand potential profile with the more commonly used term free energy profile (FEP). Such a free energy profile can serve as input for transition state theory to extract hopping frequencies, which in turn allow the extraction of diffusivities with the hopping model. We start by considering a general collective variable :math:`Q(\boldsymbol{r})` that solely depends on the center of mass position of the diffusing molecule, as that is sufficient to track the diffusion trajectory of a molecule. Our goal is to compute the grand potential :math:`\Omega(q|\mu)` for the macrostate in which at least one of the adsorbed molecules has a CV value of :math:`Q(\boldsymbol{r})=q` for given thermodynamic conditions of the chemical potential (as well as temperature and unit cell volume). To this end, we first define :math:`n(q|\mu)` as the average number of guest molecules that have a CV value of :math:`Q(\boldsymbol{r})=q`, as well as the probability :math:`p(q|\mu)` of finding a guest molecule with a CV value of :math:`Q(\boldsymbol{r})=q`. One can readily write down the following expressions for these properties:

.. math::

   \begin{aligned}
       n(q|\mu)&=\left<\sum_{k=1}^{N}\delta(Q(\vec{r}_k)-q)\right> \label{eqn:general_n_def}\\
       p(q|\mu)&=\left<\frac{1}{N}\sum_{k=1}^{N}\delta(Q(\vec{r}_k)-q)\right>.\end{aligned}

Herein, :math:`\left\langle \cdots \right\rangle` denotes the the ensemble average in the grand canonical ensemble. Herein, we will always explicitly denote the functional dependency of ensemble averages in the grand canonical ensemble towards the chemical potential :math:`\mu`, but we will leave out their dependency towards unit cell volume :math:`V` and temperature :math:`T`. The contribution :math:`Z(q|\mu)` to the total grand canonical partition function, corresponding to the macrostate where at least one guest molecule has CV value :math:`Q(\boldsymbol{r})=q` is given by:

.. math::

   \begin{aligned}
       Z(q|\mu) &= \sum_{N=1}^{+\infty} \frac{e^{\beta\mu N}}{N!\Lambda^{3N}} \int \delta\left(Q(\vec{r}_1)-q\right)e^{-\beta V(\vec{r}^N)}d\vec{r}^N.\end{aligned}

This expression seems to only track the state of the first molecule, however, given the identical and indistinguishable nature of the particles, this expression is invariant if we change :math:`\vec{r}_1` for any of the other particles present. Furthermore, notice that the summation over :math:`N` starts at :math:`1`, because in order to compute the :math:`Q` value of a guest molecule, at least one molecule needs to be present, the implications of this are outlined in the supporting information. In order to be able to compute the free energy profile of interest, i.e. \ :math:`\Omega(q|\mu)`, using cDFT, we first need to calculate the profiles :math:`n(q|\mu)` and :math:`p(q|\mu)`. Calculating :math:`n(q|\mu)` from cDFT calculations will be straightforward, however, calculating :math:`p(q|\mu)` is less trivial but it is crucial for constructing the required free energy profiles. We first consider the ensemble average :math:`F(\mu)` of a general one-particle observable :math:`\hat{f}(\vec{r})` defined as:

.. math::

   \begin{aligned}
       F(\mu) &= \left\langle \sum_{k=1}^N \hat{f}\left(\vec{r}_k\right) \right\rangle = \int \hat{f}\left(\vec{r}\right)\rho\left(\vec{r}~|\mu\right)d\vec{r},
       \label{eqn:general_F_def}\end{aligned}

where the second equation follows after some calculus that we include in the supporting information. Next, a related ensemble average is considered, which can be interpreted as the average value of :math:`\hat{f}(\vec{r})` per particle:

.. math::

   \begin{aligned}
       f(\mu) &= \left\langle \frac{1}{N}\sum_{k=1}^N \hat{f}\left(\vec{r}_k\right) \right\rangle \end{aligned}

After some derivations (see supporting information) the following relation can be found:

.. math::

   \begin{aligned}
       f(\mu) &= f_0e^{\beta\left[\Omega(\mu)-\Omega_0\right]} + \beta\int_{\mu_0}^{\mu}F(\tilde{\mu})e^{-\beta\left[\Omega(\tilde{\mu})-\Omega(\mu)\right]}d\tilde{\mu}\end{aligned}

Herein, :math:`\Omega(\mu)` represents the grand potential at the given chemical potential, while :math:`f_0` and :math:`\Omega_0` represent the values of :math:`f` and :math:`\Omega` for :math:`\mu=\mu_0`. If we now choose :math:`\mu_0\rightarrow -\infty`, then we know :math:`N_0\rightarrow 0` and therefore, :math:`\Omega_0 \rightarrow 0` (given that :math:`\Omega` encodes the degrees of freedom of the guest particles and if there are no guest particles, then there is only a single state so :math:`Z=1` and :math:`\Omega=0`). This is the ideal gas regime, and therefore :math:`f_0` corresponds to the ideal gas value of :math:`f`. Hence, we find:

.. math::

   \begin{aligned}
       f(\mu) &= e^{\beta\Omega(\mu)}\left[f_0 + \beta\int_{-\infty}^{\mu}F(\tilde{\mu})e^{-\beta\Omega(\tilde{\mu})}d\tilde{\mu}\right] \label{eqn:general_f_prop}\end{aligned}

This can now be applied to find expressions for :math:`n(q|\mu)` and :math:`p(q|\mu)`. By comparing Eq. \ `[eqn:general_n_def] <#eqn:general_n_def>`__ with Eq. \ `[eqn:general_F_def] <#eqn:general_F_def>`__ we set :math:`f(\vec{r}_k)=\delta\left(q(\vec{r}_k)-q\right)` to arrive at:

.. math::

   \begin{aligned}
       n(q|\mu) &= \int \delta\left(Q(\vec{r})-q\right)\rho\left(\vec{r}~|\mu\right)d\vec{r} \label{eq:proj_dens}\end{aligned}

This quantity can be extracted from the cDFT adsorption density profile :math:`\rho(\vec{r}|\mu)` at a given chemical potential. Furthermore, applying Eq. \ `[eqn:general_f_prop] <#eqn:general_f_prop>`__ with :math:`f(\mu)=p(q|\mu)` and :math:`F(\mu)=n(q|\mu)` and accounting for :math:`p(q;-\infty)=0` for all values of q, we find:

.. math::

   \begin{aligned}
       p(q|\mu) &= \beta\int_{-\infty}^{\mu}n(q|\tilde{\mu})e^{-\beta[\Omega(\tilde{\mu})-\Omega(\mu)]}d\tilde{\mu}\end{aligned}

In the supporting information, we also derive the relation :math:`p(q|\mu) = \frac{Z(q|\mu)}{Z(\mu)}`, which allows us to rewrite the previous equation further as:

.. math::

   \begin{aligned}
       Z(q|\mu) &= \beta\int_{-\infty}^{\mu}n(q|\tilde{\mu})e^{-\beta\Omega(\tilde{\mu})}d\tilde{\mu}\end{aligned}

and the grand potential as:

.. math::

   \begin{aligned}
    \label{eq:final_grand_pot}
       \Omega(q|\mu) &= -k_BT\ln(Z(q|\mu))\\
       &= -k_BT\ln\left[\beta\int_{-\infty}^{\mu}n(q|\tilde{\mu})e^{-\beta\Omega(\tilde{\mu})}d\tilde{\mu}\right].\end{aligned}

This grand potential is fundamentally different from the free energy profiles obtained from enhanced molecular dynamics simulations with umbrella sampling, as it is obtained in different ensembles. Here the number of particles is allowed to change at a defined chemical potential, while the framework is assumed rigid. Whereas the free energy profiles obtained from the umbrella sampling MD are calculated at a fixed number of particles, but fully accounting for the flexibility of the framework.

.. _sec:framework_flex:

Framework flexibility
=====================

Many metal-organic frameworks (MOFs) are structurally flexible. Breathing, swelling, gate opening, and linker rotation can be induced by pressure, temperature, or adsorption. For example, MIL-53 can switch between large- and narrow-pore phases, while linker rotation in ZIF-8 changes its effective aperture and therefore its adsorption and diffusion behavior.

Standard GCMC usually treats the framework as rigid. Flexibility can instead be represented by sampling multiple framework configurations, explicitly perturbing the framework during GCMC, or combining molecular dynamics with transition-matrix Monte Carlo. In cDFT, related strategies include averaging cDFT results over framework snapshots and coupling adsorption stress to a reduced deformation coordinate. The method below provides a cDFT analogue of a multi-configuration or dual-structure treatment, a full derivation and more complete description can be found in the Ph.D. thesis of Vic De Ridder.

.. _sec:cdft_ext_frameflex:

Collective-variable description
-------------------------------

The full framework configuration is denoted by :math:`\boldsymbol R`, with empty-host energy :math:`U_h(\boldsymbol R)`. Since sampling every framework degree of freedom is generally impractical, flexibility is reduced to one collective variable (CV), :math:`S(\boldsymbol R)=s`. The CV is assumed to capture the structural changes that significantly affect the external potential and the equilibrium guest density. Configurations with the same :math:`s` are therefore treated as having similar adsorption behavior.

For a fixed :math:`s`, the empty-host Helmholtz free-energy profile is

.. math:: F_h(s)=-k_BT\ln Z_h(s),

where :math:`Z_h(s)` is the host partition function restricted to configurations satisfying :math:`S(\boldsymbol R)=s`. The guest system has a corresponding constrained grand potential :math:`\Omega_g(\mu,s)`, obtained from cDFT using one of the approximations below. The total constrained grand potential is

.. math:: \Omega(s|\mu)=F_h(s)+\Omega_g(\mu,s).

The unconstrained system is obtained by integrating over the CV:

.. math:: \Omega(\mu)=-k_BT\ln\int\exp[-\beta\Omega(s|\mu)]\,ds.

In the saddle-point approximation this becomes

.. math::

   \label{eq:minimal_total_pot}
       \Omega(\mu)\approx\min_s\left[F_h(s)+\Omega_g(\mu,s)\right],

where the minimizing value :math:`s_0` can depend on chemical potential and temperature. Thus, adsorption can shift the framework toward a different structural state.

The equilibrium density is a probability-weighted average of the densities at fixed CV:

.. math::

   \label{eq:rho_flexible_integration}
       \rho(\boldsymbol r|\mu)=\int p(s|\mu)\rho(\boldsymbol r|\mu,s)\,ds,

with

.. math:: p(s|\mu)=\exp\{-\beta[\Omega(s|\mu)-\Omega(\mu)]\}.

The saddle-point approximation reduces this to :math:`\rho(\boldsymbol r|\mu)\approx\rho(\boldsymbol r|\mu,s_0)`. For a rigid framework, the CV has only one relevant value and the flexible formulation reduces to ordinary fixed-geometry cDFT.

Approximating the constrained external potential
------------------------------------------------

The exact constrained guest-framework contribution involves the canonical average

.. math:: \left\langle e^{-\beta V(\boldsymbol r^N,\boldsymbol R)}\right\rangle_{S=s},

which generally cannot be represented by a one-body potential because framework fluctuations correlate the potentials at different guest positions. Three practical approximations are available:

Approach 1: Single representative geometry.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each CV value, choose a representative structure :math:`\boldsymbol R_0(s)`, for example from constrained MD or biased minimization, and solve one cDFT problem using :math:`v(\boldsymbol r,\boldsymbol R_0(s))`. This is appropriate when fluctuations within a constrained ensemble are small and one geometry captures its external-potential landscape.

Approach 2: Effective-potential averaging.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sample :math:`N_s` framework configurations from the constrained Boltzmann distribution at fixed :math:`s` and define

.. math:: v_\text{eff}(\boldsymbol r,s)=-k_BT\ln\left[\frac{1}{N_s}\sum_{i=1}^{N_s}e^{-\beta v(\boldsymbol r,\boldsymbol R_i)}\right].

A single cDFT calculation is then performed with :math:`v_\text{eff}`. This approach neglects residual correlations between potentials at different positions, but accounts for fluctuations through a free-energy average rather than an arithmetic average of energies.

Approach 3: Density averaging.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Perform a separate cDFT calculation for each sampled structure and average the resulting densities. In the most general form, the densities are weighted by their fixed-geometry guest partition functions; if these partition functions are similar, a uniform average is sufficient:

.. math::

   \rho(\boldsymbol r|\mu,s)\approx\frac{1}{N_s}\sum_{i=1}^{N_s}
       \rho(\boldsymbol r|\mu,\boldsymbol R_i).

The approaches are not generally equivalent because the cDFT map from external potential to density is nonlinear. Approach 1 is the :math:`N_s=1` limit and is cheapest, Approach 2 retains fluctuations in an effective potential and requires one cDFT calculation per :math:`s`, while Approach 3 retains separate cDFT responses and requires multiple calculations per :math:`s`. All three coincide in the rigid-framework limit.

.. _sec:flexible_fep:

Guest free-energy profiles for flexible frameworks
--------------------------------------------------

The same framework can be used to calculate a guest free-energy profile while allowing the host to change state. Let :math:`Q(\boldsymbol r)=q` describe the guest macrostate of interest, such as position along a diffusion pathway. At fixed :math:`s`, solve cDFT to obtain :math:`\rho(\boldsymbol r|\mu,s)` and :math:`\Omega_g(\mu,s)`, then calculate the one-dimensional number density

.. math:: n(q|\mu,s)=\int\delta[Q(\boldsymbol r)-q]\rho(\boldsymbol r|\mu,s)\,d\boldsymbol r.

The corresponding guest free-energy contribution can be reconstructed from the chemical-potential integral

.. math::

   \Omega_g(q|\mu,s)=-k_BT\ln\left[\beta\int_{-\infty}^{\mu}
       n(q|\tilde\mu,s)e^{-\beta\Omega_g(\tilde\mu,s)}\,d\tilde\mu\right].

To include framework flexibility, combine the guest and host contributions before integrating over :math:`s`:

.. math::

   e^{-\beta\Omega(q|\mu)}=
       \int e^{-\beta[F_h(s)+\Omega_g(q|\mu,s)]}\,ds.

A saddle-point estimate is obtained by minimizing the bracketed quantity with respect to :math:`s`. The final profile can alternatively be obtained by first calculating the full flexible density using Eq. \ `[eq:rho_flexible_integration] <#eq:rho_flexible_integration>`__ and then projecting it onto :math:`q`:

.. math::

   \label{eq:therm_int}
       n(q|\mu)=\int\delta[Q(\boldsymbol r)-q]\rho(\boldsymbol r|\mu)\,d\boldsymbol r.

Practical workflow
------------------

The flexible-cDFT procedure is therefore:

#. Choose a collective variable :math:`S(\boldsymbol R)` and obtain the empty-host free-energy profile :math:`F_h(s)`.

#. Select representative geometries or constrained framework snapshots for each :math:`s`.

#. Use Approach 1, 2, or 3 to calculate :math:`\rho(\boldsymbol r|\mu,s)` and :math:`\Omega_g(\mu,s)`.

#. Combine :math:`F_h(s)` and :math:`\Omega_g(\mu,s)` to determine the equilibrium framework state or integrate over all :math:`s` values.

#. For diffusion or other guest macrostates, project the density onto :math:`Q` and apply the same flexibility weighting.

This workflow is depicted in the schematic below.

.. figure:: /_static/images/methodology.png
   :alt: Workflow for calculating guest free-energy profiles in flexible host materials using cDFT, illustrated for ethane in ZIF-8.
   :width: 100.0%