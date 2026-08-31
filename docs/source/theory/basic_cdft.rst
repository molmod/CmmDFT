.. _sec:cDFT:

Basic classical density functional theory
=========================================

Classical density functional theory (cDFT) describes an inhomogeneous fluid using the one-body number-density profile :math:`\rho(\mathbf{r})`. Instead of tracking the position of every molecule, cDFT determines the most probable equilibrium density by minimizing a thermodynamic potential.

.. _sec:density:

Density profile
---------------

For a mixture, the density is a vector containing the density of each species:

.. math::

   \rho(\mathbf{r})
   =
   \left\{
   \rho_1(\mathbf{r}),\rho_2(\mathbf{r}),\ldots,\rho_s(\mathbf{r})
   \right\},

where :math:`s` is the number of species. The average number of particles of species :math:`i` is

.. math::

   N_i =
   \int \rho_i(\mathbf{r})\,\mathrm{d}\mathbf{r}.

For a single-component fluid, the species index can be omitted.

Grand potential
---------------

CmmDFT uses the grand-canonical ensemble, in which the temperature, chemical potential, and volume are fixed. The grand-potential functional is

.. math::

   \Omega[\rho]
   =
   F[\rho]
   +
   \sum_i\int
   \rho_i(\mathbf{r})
   \left[
      V_{\mathrm{ext},i}(\mathbf{r})-\mu_i
   \right]
   \,\mathrm{d}\mathbf{r},

where :math:`F` is the intrinsic Helmholtz free-energy functional,
:math:`V_{\mathrm{ext},i}` is the external potential acting on species :math:`i`, and :math:`\mu_i` is its chemical potential.

For a system without an external potential, this reduces to

.. math::

   \Omega[\rho]
   =
   F[\rho]
   -
   \sum_i\mu_iN_i.

The equilibrium density profiles minimize the grand potential:

.. math::

   \Omega[\rho] =
   \min_{\rho}\Omega[\rho].

Euler--Lagrange equation
------------------------

The equilibrium profiles satisfy the Euler--Lagrange equations

.. math::

   \frac{\delta\Omega}{\delta\rho(\mathbf{r})}=0.

Substitution of the grand-potential functional gives

.. math::

   \frac{\delta F}{\delta\rho(\mathbf{r})}
   +
   V_{\mathrm{ext}}(\mathbf{r})
   -
   \mu
   =
   0.

Equivalently, the chemical potential at equilibrium is

.. math::

   \mu =
   \frac{\delta F}{\delta\rho(\mathbf{r})}
   +
   V_{\mathrm{ext}}(\mathbf{r}).

This equation is solved numerically to obtain the equilibrium density profile.

Free-energy decomposition
--------------------------

The intrinsic Helmholtz free-energy functional is decomposed into an ideal and an excess contribution:

.. math::

   F[\rho]
   =
   F_{\mathrm{id}}[\rho]
   +
   F_{\mathrm{ex}}[\rho].

The excess contribution describes intermolecular interactions and is discussed in :doc:`excess_functionals`.

Ideal-gas contribution
----------------------

The ideal-gas contribution is known exactly:

.. math::

   F_{\mathrm{id}}[\rho]
   =
   k_{\mathrm{B}}T
   \sum_i\int
   \rho_i(\mathbf{r})
   \left[
      \ln\left(\Lambda_i^3\rho_i(\mathbf{r})\right)-1
   \right]
   \,\mathrm{d}\mathbf{r},

where :math:`k_{\mathrm{B}}` is the Boltzmann constant, :math:`T` is the temperature, and :math:`\Lambda_i` is the thermal de Broglie wavelength of species :math:`i`.

Its functional derivative is

.. math::

   \frac{\delta F_{\mathrm{id}}}
   {\delta\rho_i(\mathbf{r})}
   =
   k_{\mathrm{B}}T
   \ln\left(\Lambda_i^3\rho_i(\mathbf{r})\right).

Using the ideal and excess contributions, the Euler--Lagrange equation becomes

.. math::

   \rho_i(\mathbf{r})
   =
   \Lambda_i^{-3}
   \exp\left[
      \beta\mu_i
      -
      \beta V_{\mathrm{ext},i}(\mathbf{r})
      -
      \beta
      \frac{\delta F_{\mathrm{ex}}}
      {\delta\rho_i(\mathbf{r})}
   \right],

where

.. math::

   \beta=\frac{1}{k_{\mathrm{B}}T}.

This expression forms the basis of the iterative solvers used by CmmDFT.

External potentials
-------------------

The external potential represents interactions between the fluid and its surroundings, such as a solid framework, wall, or confining surface. It can also include imposed fields. Regions where the external potential is very large are inaccessible to the fluid, causing the corresponding density to approach zero.

Bulk limit
----------

For a homogeneous system, the density is independent of position:

.. math::

   \rho_i(\mathbf{r})=\rho_i^{\mathrm{bulk}}.

The external potential is constant or zero, and the functional equations reduce to the bulk thermodynamic relations used by the equation of state. These bulk properties provide the chemical potentials and reference densities required for inhomogeneous calculations.

