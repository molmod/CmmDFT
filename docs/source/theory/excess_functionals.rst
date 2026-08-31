.. _sec:excess_functional:

Approximations to the excess functional
=======================================

The excess free energy functional is not known exactly, finding accurate approximations is one of the most important challenges in cDFT. Many approximations have already been proposed in literature, each with their strengths and weaknesses. In this section, we will discuss some of the most commonly used excess functionals.

.. _sec:FMT:

Fundamental measure theory
--------------------------

To approximate the excess free energy functional, it is often divided into several contributions. The most common division is in a hard sphere repulsive contribution and an attractive contribution:

.. math:: \mathcal{F}^{ex}[\rho] = \mathcal{F}^{hs}[\rho] + \mathcal{F}^{attr}[\rho].

The hard sphere repulsion models the fact that two particles cannot occupy the same volume, it is most commonly modelled with fundamental measure theory (FMT). FMT was developed by Rosenfeld as a means of calculating the excess free energy of a mixture of hard sphere particles.  :cite:`rosenfeld1989free` The hard sphere potential is given as:

.. math::

   \begin{aligned}
   {1}
       w_{ij}(r) = & \begin{cases}
       \infty & r\leqslant R_i + R_j \\
       0 & r>R_i + R_j
       \end{cases}\end{aligned}

Here :math:`R_i` and :math:`R_j` are the radii of the hard sphere for species :math:`i` and :math:`j`, which are estimated from the true interaction potential between the particles. Barker and Henderson developed a statistical mechanics model for the interaction of molecules consisting of a hard sphere reference augmented with a perturbation series in terms of the attractive interaction. They estimated the hard-sphere radius from the true interaction potential :math:`v(r)` as follows :cite:`barker1967perturbation`

.. math:: R_i = \frac{1}{2} \int_{0}^{\sigma_i} \left( 1-e^{-\beta v(r)} \right) dr,

When the intermolecular interaction is described by means of a Lennard-Jones potential, this hard sphere radius can be approximated with the following empirical formula :cite:`cotterman1986molecular`

.. math::

   \label{eq:hs_d}
       R_i = \frac{1}{2} \sigma_i\left(\frac{1+0.29777T^*}{1+0.33163T^*+0.0010477T^{*2}}\right),

where :math:`T^*=k_BT/\epsilon_i` is the reduced temperature.

The excess functional for a fluid of hard spheres is first derived in a one dimensional situation, where the spheres are replaced with hard rods, which is then generalized to three dimensions. Much of the following discussion is based on a review paper by Roth. :cite:`roth2010fundamental`

The central idea of FMT is to express the excess free energy as a functional of weighted densities instead of the raw local density :math:`\rho(\vec{r})`. These are convolutions of local densities and geometric weight functions. In the one dimensional case these weighted densities can be written as:

.. math::

   \label{eq:WeightDens}
       n_{\alpha}(z) = \sum_i^{nc}\int dr'\rho_i(z')\omega^i_{\alpha}(z-z'),

where :math:`\rho_i` is the density of gas species :math:`i` and :math:`nc` is the number of component species. Hence, they are indeed convolutions with weight functions :math:`\omega^i_{\alpha}`. In one dimension two weight function are used representing the surface and the volume of the rod respectively:

.. math::

   \begin{aligned}
       \omega_0^i(z) &= \frac{1}{2}(\delta(z-R_i)+\delta(z+R_i)), \quad
       \omega_1^i(z) = \Theta(R_i-z).\end{aligned}

In these expressions the Dirac delta function (:math:`\delta(z)`) and the Heavyside step function (:math:`\Theta(z)`) are used. The excess free energy in 1-D takes the form:

.. math::

   \label{eq:Ex1D}
       \beta \mathcal{F}^{hs}[\{\rho_i\}] = \int dr \Phi(\{n_{\alpha}\}).

Now the :math:`\Phi` functional needs to be defined, the key ingredient connecting geometry to thermodynamics is the Mayer-f function:

.. math::

   \begin{aligned}
   {1}
       f_{ij}(z) = e^{-\beta w_{ij}(z)}-1 = \begin{cases}
       -1 & z\leqslant R_i + R_j \\
       0 & z>R_i + R_j
       \end{cases}\end{aligned}

For hard sphere repulsion this function has a simple geometric interpretation, it defines the excluded volume, this being the region in which the centre of one particle cannot be placed, relative to another spherical particle. This volume is a sphere with radius :math:`R_i+R_j`, while the interpretation for hard rods is equivalent. For one dimension, this function decomposes as:

.. math:: -f_{ij}(z)=\omega_1^i\otimes\omega^j_0+\omega_0^i\otimes\omega^j_1,

where the convolutions are written as:

.. math:: \omega_1^i\otimes\omega_0^j(z_i-z_j) =\int dz'\omega_1^i(z'-z_i)\omega_0^j(z'-z_j).

At low densities the excess energy for mixtures interacting with pairwise potentials can be expanded as:

.. math::

   \begin{aligned}
   \label{eq:LowDens}
       \beta \mathcal{F}^{hs}[\{\rho_i\}] &= -\frac{1}{2}\sum_{i,j}\int dz_1\int dz_2 \rho_i(z_1)\rho_j(z_2)f_{ij}(z_{12}) + O(\rho^3).\end{aligned}

To extend the theory to three dimensions it is instructive to see that the excluded volume for two convex bodies can be written as :cite:`evans1992density`

.. math:: V_{ij} = V_i + R_iS_j + S_iR_j + V_j,

where :math:`V_k`, :math:`S_k` and :math:`R_k` are the volume, the surface and the mean radius of curvature of the body k respectively. The volume of the joined cluster can be written as a function of the geometrical measures of the individual particles. Now in three dimensions the Mayer-f function can be decomposed to:

.. math::

   \label{eq:decon}
       -f_{ij}(r)=\omega_3^i\otimes\omega^j_0+\omega_0^i\otimes\omega^j_3 + \omega_2^i\otimes\omega^j_1 + \omega_1^i\otimes\omega^j_2 - \vec{\omega}_2^i\otimes\vec{\omega}^j_1 -\vec{\omega}_1^i\otimes\vec{\omega}^j_2,

where the weight functions are defined as:

.. math::

   \begin{aligned}
       \omega^i_3(\vec{r}) &= \Theta(R_i-|\vec{r}|)& \omega^i_2(\vec{r}) &= \delta(R_i-|\vec{r}|)&    \vec{\omega}^i_2(\vec{r}) &= \frac{\vec{r}}{|\vec{r}|}\delta(R_i-r)\\
       \omega^i_1(\vec{r}) &= \frac{\omega^i_2(\vec{r})}{4\pi R_i}
       & \omega^i_0(\vec{r}) &= \frac{\omega^i_2(\vec{r})}{4\pi R_i^2}&
       \vec{\omega}^i_1(\vec{r}) &= \frac{\vec{\omega}^i_2(\vec{r})}{4\pi R_i}.\end{aligned}

These weights represent the fundamental Minkowski functionals of the individual particles. Specifically, spatial integration of the scalar weight functions :math:`\omega_{\alpha}^i` yields the core geometric properties of particle :math:`i`: the volume :math:`V_i` (for :math:`\alpha=3`), the surface area :math:`S_i` (for :math:`\alpha=2`), the mean integral curvature radius :math:`R_i` (for :math:`\alpha=1`), and the dimensionless Euler characteristic :math:`\chi_i` (for :math:`\alpha = 0`). The Euler characteristic is a topological invariant that characterizes the shape and connectivity of a body regardless of its size; for the solid, convex hard spheres considered here, it simplifies to :math:`\chi_i = 1`, effectively serving as a topological indicator for particle number density.

The vector weight functions arise from the broken spatial symmetry encountered in inhomogeneous environments. While an isolated hard sphere is geometrically isotropic, a non-uniform distribution of surrounding particles, such as that found near a solid boundary or within a tight confinement in a nanoporous materials, induces a highly directional, anisotropic environment of excluded volume interactions. The vector weights track these localized structural gradients by acting as surface normal indicators. In uniform bulk fluids, the surrounding particle distribution is isotropic, causing these vector contributions to cancel out exactly by directional symmetry, yielding :math:`\vec{n}_2=0`.

Rosenfeld proposed an ansatz for the excess free-energy functional based on the 1-D case from earlier:

.. math:: \beta\mathcal{F}^{hs}[\{\rho_i\}]=\int\Phi(\{n_{\alpha}(\vec{r})\})d\vec{r},

with :math:`\Phi` being the excess free energy density, which is again a function of the weighted densities. :math:`\Phi` is determined by dimensional analysis and the requirement that the correct low density limit (i.e., the Mayer-\ *f* decomposition) is recovered. This means that the functional should reduce to the known result when the system is confined to a lower-dimensional geometry, this concept is called dimensional crossover. For example, a 3D hard-sphere fluid compressed into a cavity that holds at most one particle should produce the exact 0D partition function.

By organizing the geometric terms according to their scaling behavior with respect to the local packing fraction :math:`n_3`, the dimensionless energy density is written as:

.. math::

   \label{eq:rosenfeld}
       \Phi = \phi_1(n_3)n_0 + \phi_2(n_3)(n_1n_2-\vec{n}_1\cdot\vec{n}_2) + \phi_3(n_3)(n_2^3-3n_2\vec{n}_2^2).

The functions :math:`\phi_1,\phi_2, \phi_3` are fixed by using an equation of state (EOS) as a reference for tha bulk hard-sphere fluid. For the original FMT, the Percus-Yevick (PY) EOS was used, resulting in the following functions :cite:`rosenfeld1989free`:

.. math::

   \begin{split}
       \phi_1(n_3) &= -\ln(1-n_3) \\
       \phi_2(n_3) &= \frac{1}{1-n_3} \\
       \phi_3(n_3) &= \frac{1}{24\pi n_3^2(1-n_3)^2}
   \end{split}

Later the White-Bear version was introduced, based on the more accurate Carnahan-Starling (CS) EOS. :cite:`roth2002fundamental, yu2002structures` This version uses the same :math:`\phi_1` and :math:`\phi_2`, but a different :math:`\phi_3`:

.. math:: \phi_3(n_3) = \frac{n_3+(1-n_3)^2\ln(1-n_3)}{36\pi n_3^2(1-n_3)^2}

The White-Bear Mark II version is an improvement on the previous, it is based on an improved equation of state, :cite:`hansen2006new` it uses different :math:`\phi_2` and :math:`\phi_3`\  :cite:`hansen2006density`:

.. math::

   \begin{split}
       \phi_2(n_3) &= \frac{5n_3-3n_3^2+2(1-n_3)\ln(1-n_3)}{3n_3(1-n_3)}\\
       \phi_3(n_3) &= \frac{-2(n_3-3n_3^2+n_3^3+(1-n_3)^2\ln(1-n_3))}{3n_3^2(24\pi(1-n_3)^2)}
   \end{split}

The version as presented above in equation `[eq:rosenfeld] <#eq:rosenfeld>`__ is sometimes referred to as the Rosenfeld version. This version did not accurately model the transition of the hard sphere fluid into a solid, periodic crystal. In a crystal, the density becomes highly localized around lattice sites, meaning that the vector densities become large relative to their scalar counterparts, as the localization breaks the isotropy. The original functional fails in this regime because the cubic term :math:`\phi_3(n_3)(n_2^3 - 3n_2\vec{n}_2^2)` does not correctly handle the limit where :math:`|\vec{n}_2| \to n_2`, i.e. where the vector density approaches the magnitude of the scalar surface density. Physically, this limit corresponds to a highly non-uniform situation, caused by sharply localized densities. In this limit the original functional diverges unphysically, producing incorrect free energies for the solid phase. These strong anisotropies also occur for fluids in strong confinement, i.e. gases adsorbed in nanoporous materials, therefore treating this contribution is essential for obtaining accurate adsorption results at high loadings.

The anti-symmetrical version was introduced :cite:`rosenfeld1996dimensional, rosenfeld1997fundamental` to solve this issue. It replaces the cubic term with:

.. math:: \phi_3(n_3)(n_2^3-3n_2\vec{n}_2^2) \rightarrow \phi_3(n_3)\left(n_2^3\left(1-\frac{\vec{n}_2^2}{n_2^2}\right)^3\right),

The ratio\ :math:`|\vec{n}_2|/n_2` acts as a local measure of anisotropy: it is zero in a uniform fluid and approaches one near a fully localized density peak. Raising this to the third power ensures the entire cubic contribution vanishes smoothly as localization becomes extreme, which removes the unphysical divergence. The original Rosenfeld functional is recovered as the first two terms of the binomial expansion, the anti-symmetrical version is therefore consistent at low anisotropy.

To solve this issue more systematically Tarazona introduced tensorial weight functions. :cite:`tarazona2000density` While the antisymmetric functional improves the description of freezing, it does not fully resolve the underlying problem. The vector densities carry directional information, but they cannot describe the full shape of that anisotropy. Near a lattice site, the density is peaked along multiple crystallographic directions simultaneously, creating a structure that a single vector cannot capture. The tensorial weight function is given as :cite:`tarazona2000density`:

.. math:: w_{ij}(\vec{r}) = \left( \frac{r_{i}r_{j}}{R^2}- \frac{\delta_{ij}}{3}\right)\delta(R-|\vec{r}|)

This is a rank-2 tensor defined on the surface of the sphere of radius :math:`R`. At each surface point :math:`\vec{r}`, it captures the shape of the surface geometry. The subtraction of :math:`\delta_{ij}/3` ensures the weight function is traceless, so it is insensitive to isotropic compression and responds only to genuine shape asymmetry. The tensorial weighted density is then given as:

.. math:: T_{ij}(\vec{r}) = \int d\vec{r}'\rho(\vec{r}')w_{ij}(\vec{r}-\vec{r}').

The tensorial contribution can be written as a correction :cite:`oettel2010free,bernet2020tensorial`

.. math:: \Phi^{FMT}_T = \frac{9\phi_3(n_3)}{2}\left(\vec{n}_2\vec{T}\vec{n}_2  - \text{Tr}(\vec{T}^3)\right).

Mean field approximation
------------------------

While FMT provides an accurate description of hard-sphere repulsion, the attractive part of the excess free energy requires a separate treatment. A natural and computationally convenient choice is the mean field approximation (MFA), which accounts for attractive interactions while deliberately ignoring the short-range correlations between particle positions. We begin with the formally exact expression for the two-body attractive interaction energy :cite:`evans1992density`

.. math:: \mathcal{F}^{\text{MFA}}[\rho] = \frac{1}{2}\sum_{i}^{nc}\sum_{j}^{nc}\iint\rho_{ij}(\vec{r},\vec{r}')w_{ij}(|\vec{r}-\vec{r}'|)\,\mathrm{d}\vec{r}\mathrm{d}\vec{r}'.

In this expression, :math:`w_{ij}(|\vec{r}-\vec{r}'|)` represents the pairwise interaction potential between species :math:`i` and :math:`j`. The pair density function, :math:`\rho_{ij}(\vec{r},\vec{r}')`, captures the probability of simultaneously finding a particle of species :math:`i` at position :math:`\vec{r}` and a particle of species :math:`j` at position :math:`\vec{r}'`. This function can be decoupled as:

.. math:: \rho_{ij}(\vec{r},\vec{r}') = \rho_i(\vec{r})\rho_j(\vec{r}')g_{ij}(\vec{r},\vec{r}'),

where :math:`g_{ij}(\vec{r},\vec{r}')` is the pair correlation function, which encodes how the presence of a particle of species :math:`i` modulates the local probability density of finding a particle of species :math:`j` nearby. In an ideal gas, :math:`g_{ij}` equals unity everywhere, denoting entirely uncorrelated particle positions.

In real fluids, local cohesive forces induce structural correlations, meaning :math:`g_{ij}` deviates significantly from unity. The mean field approximation simplifies this behavior by assuming particles are completely uncorrelated, except for the fact that they cannot overlap:

.. math::

   \label{eq:mfa_g}
       g_{ij}(\vec{r},\vec{r}') = \begin{cases}
       0 & |\vec{r}-\vec{r}'|<\sigma_{ij}, \\
       1 & \text{elsewhere},
       \end{cases}

where :math:`\sigma_{ij}` is the distance at which the interatomic potential switches from repulsive to attractive (:math:`w_{ij}(\sigma_{ij})=0`). This approximation of :math:`g_{ij}` as a step-function ensures that the MFA contribution isolates purely long-range attractions, preventing the double-counting of short-range repulsions already handled by the FMT framework. Substituting Equation \ `[eq:mfa_g] <#eq:mfa_g>`__ into the energy functional yields:

.. math::

   \label{eq:MFA}
       \mathcal{F}^{\text{MFA}}[\rho] = \frac{1}{2}\sum_{i}^{nc}\sum_{j}^{nc}\iint\rho_i(\vec{r})\rho_j(\vec{r}')w_{ij}^{\text{attr}}(|\vec{r}-\vec{r}'|)\,\mathrm{d}\vec{r}\mathrm{d}\vec{r}',

where :math:`w_{ij}^{\text{attr}}(|\vec{r}-\vec{r}'|)` is the truncated attractive part of the interaction potential:

.. math::

   w_{ij}^{\text{attr}}(r) = \begin{cases}
       0 & r<\sigma_{ij}, \\
       w_{ij}(r) & \text{else}.
       \end{cases}

The resulting MFA functional is highly efficient, straightforward to implement and surprisingly effective. :cite:`archer2017standard` However, by setting :math:`g_{ij}(r) = 1` in the attractive region (:math:`r \ge \sigma_{ij}`), the formulation systematically underestimates the magnitude of the attractive interactions. :cite:`tang2004modeling` In a physical fluid, attractive forces cause molecules to preferentially cluster within their mutual potential energy wells, resulting in a local coordination shell where :math:`g_{ij}(r) > 1`. Because the MFA neglects this local density enhancement adjacent to a central particle, it inherently underestimates the total cohesive energy of the fluid. In cDFT applications, this systematic underestimation typically manifests as a deficit in predicted adsorption capacities at elevated pressures or lower temperatures where local molecular correlations become dominant.

Local density approximation
---------------------------

The local density approximation (LDA) is the most simple approximation for the excess functional, it is also well known within the context of electronic DFT. The basic idea is to convert a model valid for homogeneous systems into a form, applicable to inhomogeneous systems, in which the excess free energy density at a given point only depends on the local density, i.e. the density at that same point. It is given as follows:

.. math::

   \label{eq:FMT}
       \mathcal{F}^{LDA} = \int f^{ex}_{V}(\rho(\vec{r})) d\vec{r}.

The volume is divided in infinitesimal parts :math:`d\vec{r}` in which the excess free energy per volume :math:`f^{ex}_{V}(\rho(\vec{r}))` as a function of the density is given by an EOS of choice. Popular choices of this EOS are the Modified Benedict-Webb-Rubin (MBWR) EOS for a Lennard-Jones fluid :cite:`johnson1993lennard` or the Carnahan-Starling EOS for a hard sphere fluid. :cite:`carnahan1969equation` The weakness of this functional is given by its neglect of spatial correlations between different volume units. This makes LDA unsuited for situations where the density has large fluctuations. This is the case for gas adsorption in MOFs, therefore this approximation is not used in this study.

Weighted density approximation
------------------------------

The weighted density approximation (WDA) is an improvement on LDA because it includes some correlation by using a weighted density, it is nonlocal as the weighted density is an average over the volume surrounding a point, different from the local density. The introduction of this weighted density avoids singularities in calculations and can be used for systems with strong inhomogeneity. This method was first proposed by Tarazona as a functional which can account for both ’local’ thermodynamics and correlations between the particles. :cite:`tarazona1984simple` The functional is given as follows:

.. math::

   \begin{aligned}
    \label{eq:WDAV}
       \mathcal{F}^{\text{WDA}} &= \int f^{ex}_{V}(\bar{\rho}(\vec{r}))d\vec{r}\\
       \bar{\rho}(\vec{r}) &= \int\rho(\vec{r}')\omega(\vec{r}-\vec{r}')d\vec{r}'.\end{aligned}

Here :math:`f^{ex}_{V}` is the excess free energy per volume as given by an EOS. The level of included correlation can now be determined by choosing the weight function :math:`\omega(\vec{r}-\vec{r}')`. For simulations of adsorption in MOFs a simple weight function is often used, in the form of a Heavyside step function:

.. math:: \omega(\vec{r}) = \frac{3}{4\pi R^3}\Theta(r - R).

This function accounts for the number of particles present in the volume associated with the size of the particle, represented by the hard-sphere radius :math:`R`. Other, more complicated weight functions can also be chosen, however, the trade-off between accuracy and efficiency has to be made carefully.

Correlation correction
----------------------

Yu Yang-Xin proposed an additional WDA contribution to the excess functional on top of MFMT and MFA, to account for correlation effects between guest particles. :cite:`yu2009novel` This WDA contribution was designed to account for the difference in equation of state of the bulk guest as predicted by cDFT with just MFMT and MFA compared to the equation of state for a Lennard-Jones fluid, i.e. the Modified Benedict-Webb-Rubin (MBWR). The WDA contribution is expressed as:

.. math:: \mathcal{F}^{\text{corr}} = k_BT\int\rho(\vec{r})f^{\text{corr}}(\bar{\rho}(\vec{r}))d(\vec{r}),

where :math:`f^\text{corr}` expresses the correlation contribution to the free energy density arising from the difference between the ’true Lennard-Jones’ equation of state (here given by MBWR) and its MFMT+MFA approximation. Herein we use the Carnahan Starling (CS) EOS for MFMT and the analytical mean field EOS:

.. math::

   \begin{aligned}
       f^{corr}(\rho) &= f^{LJ}_{b}(\rho) - f^{CS}_{b}(\rho) - f^{MFA}_{b}(\rho)\\
       f^{MFA}_{b}(\rho) &= -\frac{16}{9}\pi\beta\epsilon\rho\sigma^3\\
       f^{CS}_{b}(\eta)&=\frac{4\eta-3\eta^2}{(1-\eta)^2} \quad \mathrm{with} \quad \eta=\frac{4\pi\sigma^3}{3}\rho.\end{aligned}

Here :math:`\eta` is the packing fraction and :math:`\epsilon` and :math:`\sigma` are the Lennard-Jones parameters of the particles. This functional has been tested by various groups for adsorption of simple gases such as hydrogen :cite:`fu2015density` and methane. :cite:`fu2015classical`

Perturbed chain statistical associating fluid theory
----------------------------------------------------

The family of statistical associating fluid theory (SAFT) functionals is a prominent choice in many cDFT applications, mainly due to the success of the underlying SAFT EOS. These models are based on the first-order thermodynamic perturbation theory (TPT1) developed by Wertheim, :cite:`wertheim1984fluidsI,wertheim1984fluidsII,wertheim1986fluidsIII,ZMPITAS2016121` which describes non-spherical molecules as chains of bonded, tangent spherical segments. Based on this underlying theory, several EOS variants were developed, such as interfacial SAFT (iSAFT), :cite:`tripathi2005microstructure2` Perturbed-Chain SAFT (PC-SAFT), :cite:`gross2000application,gross2001perturbed,gross2009density` SAFT for a potential with variable range (SAFT-VR), :cite:`gil1997statistical, gil1997thermodynamics` and SAFT-\ :math:`\gamma` Mie :cite:`lymperiadis2007group, papaioannou2014group` which employs a group-contribution approach with heteronuclear Mie segments., More elaborate discussions of these models can be found in Refs.  :cite:`economou2002statistical, tan2008recent` To utilize these SAFT expressions within the context of cDFT for inhomogeneous fluids, the bulk excess free energy density is expressed by means of WDA. By substituting the local segment density with the non-local weighted density, the bulk SAFT formulations are extended to account for the spatial variations and packing effects encountered under confinement. :cite:`tripathi2005microstructure2,sauer2017classical`

In SAFT, a non-spherical molecule is represented as a chain of tangentially bonded spherical segments. In its simplest form, the segments are identical, making a homonuclear chain, and a non-associating molecule is defined by three primary parameters: the temperature-independent segment diameter :math:`\sigma`, the segment-segment interaction energy depth :math:`\epsilon`, and the number of segments per molecule :math:`m`. Additional parameters can be introduced to model complex interactions such as association sites, :cite:`li2006density` polar forces, :cite:`gross2005equation, gross2006equation, kleiner2006equation` or ionic charges. :cite:`held2014epc,shen2018developing` To account for temperature-dependent soft-core repulsions, the framework utilizes the Barker-Henderson perturbation theory :cite:`barker1967perturbation` alongside the empirical temperature-dependent effective hard-sphere diameter formula originally proposed by Chen and Kreglewski :cite:`chen1977applications`:

.. math:: d_i(T)=\sigma_i\left[1-0.12\exp\left(-\frac{3\epsilon_i}{k_BT}\right)\right].

While original formulations of SAFT apply dispersion perturbations to an unbonded hard-sphere reference system, Perturbed-Chain SAFT (PC-SAFT) fundamentally alters this baseline. :cite:`gross2000application, gross2001perturbed` In PC-SAFT, the reference system is chosen as the fully formed hard-chain fluid itself. Consequently, the attractive dispersion perturbation theory is applied directly to the entire chain molecule rather than to isolated spherical segments. When mapping PC-SAFT into the framework of cDFT, this choice dictates that the density profile :math:`\rho(\vec{r})` directly represents the localized *molecular* density rather than a density of individual, independent segments. In the framework of PC-SAFT for cDFT applications, the excess functional is typically composed of the following contributions :cite:`shen2013hybrid`

.. math:: \mathcal{F}^{ex}[\rho] = \mathcal{F}^{hs}[\rho] + \mathcal{F}^{hc}[\rho] + \mathcal{F}^{disp}[\rho]

where the contributions are respectively hard sphere, hard chain, dispersion. The hard sphere energy is often modeled using FMT (as explained above), the hard chain and dispersion contributions are explained below. Additional terms have been developed in this framework, such as an association contribution, :cite:`chapman1988phase,huang1990equation,li2006density` modeling interactive association sites on molecules, a multipolar contribution for the modeling of dipole and quadrupole interactions :cite:`tumakaka2004application,gross2005equation,gross2006equation,kleiner2006equation` or contributions for ionic interactions, :cite:`held2014epc,shen2018developing` these were not considered in this work.

Hard sphere repulsion
~~~~~~~~~~~~~~~~~~~~~

The hard sphere contribution is based on the FMT framework as described above, however the calculation of the weighted densities is slightly different. As the :math:`\sigma` and :math:`\epsilon` parameters of the molecule now represent the radius and attraction of a single segment in the chain, constructing the weighted densities as before would not give the hard sphere repulsion of the full molecule. Therefore the weighted densities of a mixture of homonuclear chains are given as :cite:`roth2010fundamental`:

.. math:: n_{\alpha} = \sum_i m_i\int d\vec{r}'\rho_i(\vec{r}')w_{\alpha}^{i}(\vec{r}-\vec{r}')d\vec{r}',

with these densities the hard sphere free energy is calculated identically as before.

Chain contribution
~~~~~~~~~~~~~~~~~~

Because the molecules are represented as connected segments, the segments must remain in contact. This restriction reduces the configurational freedom of the fluid compared to a monomer system. The chain functional models this entropic and energetic cost of the chain forming. It is derived from TPT1 in the limit of complete association, where only segment-segment contacts contribute, and was first derived for cDFT by Tripathi and Chapman  :cite:`tripathi2005microstructure1, tripathi2005microstructure2`:

.. math::

   \label{eq:ch_cont}
       \begin{split}
       \mathcal{F}^{hc}[\rho] &= \sum_i^{nc}(m_i-1)\int\rho_i(\vec{r})(\ln\rho_i(\vec{r})-1)d\vec{r}\\
        & - \sum_i^{nc}(m_i-1)\int\rho_i(\vec{r})(\ln(y^{dd}_{ij}(\{\xi_{\gamma}\})\lambda_i(\vec{r}))-1)d\vec{r}.
       \end{split}

The first term is a correction term introduced by Sauer and Gross, :cite:`sauer2017classical` due to the fact that PC-SAFT works with molecular rather than segment densities, as in the original derivation. Here :math:`y^{dd}_{ij}(\{\xi_{\gamma}\})` is the cavity correlation function between two segments, evaluated at contact distance:

.. math::

   \begin{split}
       y^{dd}_{ij}(\{\xi_{\gamma}\}) = \frac{1}{1-\xi_3}&+\frac{3\sigma_i\sigma_j}{\sigma_i + \sigma_j}\frac{\xi_2}{(1-\xi_3)^2}\\
       & + 2\left(\frac{\sigma_i\sigma_j}{\sigma_i+\sigma_j}\right)^2\frac{\xi_2^2}{(1-\xi_3)^3}.
       \end{split}

This function is evaluated with averaged densities:

.. math:: \xi_{\gamma}(\vec{r}) = \frac{\pi}{6} \sum_i m_i d_i^{\gamma}\bar{\rho}^{hc}_i(\vec{r}),

where :math:`d_i` is the hard-sphere diameter and the weighted densities are given as:

.. math:: \bar{\rho}_i^{hc}(\vec{r}) = \frac{3}{4\pi d_i^3}\int\rho_i(\vec{r}')\Theta(d_i-|\vec{r}-\vec{r}'|)d\vec{r}'.

The surface-averaged density :math:`\lambda_i` accounts for segment-segment connectivity by averaging the density over the surface of a sphere of diameter :math:`d_i`:

.. math:: \lambda_i(\vec{r}) = \frac{1}{4\pi d_i^2}\int\rho_i(\vec{r}')\delta(d_i - |\vec{r}-\vec{r}'|)d\vec{r}',

where :math:`\delta` represents the Dirac delta distribution. The contrast between :math:`\bar{\rho}_i^{hc}` (a volume average) and :math:`\lambda_i` (a surface average) reflects their different physical roles: the former characterizes the local packing environment, while the latter specifically probes the contact probability needed for the chain bonding term.

Dispersion interaction
~~~~~~~~~~~~~~~~~~~~~~

The dispersion functional accounts for the attractive van der Waals interactions between segments, it is constructed by extending the bulk PC-SAFT dispersion term to inhomogeneous systems via a weighted density approximation. Following the WDA1 formulation of Sauer and Gross, :cite:`sauer2017classical,sun2021accelerate` the functional is written as:

.. math:: \mathcal{F}^{disp}[\rho(\vec{r})] = \sum_i\sum_j\int\bar{\rho}_i(\vec{r})\bar{\rho}_j(\vec{r})f^{disp}_{ij}(\bar{\eta}(\vec{r}),\bar{m})d\vec{r},

where the local density in the integral of Equation \ `[eq:WDAV] <#eq:WDAV>`__ is replaced by the weighted density, this was found to improve the results for confined systems. The weighted density is defined as:

.. math:: \bar{\rho}^{disp}_i(\vec{r}) = \frac{3}{4\pi(\psi d_i)^3}\int\rho_i(\vec{r}')\Theta[\psi d_i - |\vec{r}'-\vec{r}|]d\vec{r}',

here :math:`\psi` is a scaling parameter, which is set to :math:`1.3862`, following the value obtained by Sauer and Gross by fitting vapor-liquid surface tensions of n-alkanes. :cite:`sauer2017classical` This parameter is introduced to incorporate the long-range dispersion interaction, therefore, a value larger than 1 is expected. The free energy contribution as a function of the density is taken from Ref.  :cite:`gross2001perturbed`. It is given by:

.. math::

   \begin{split}
       f^{disp}_{ij}(\bar{\eta}(\vec{r}),\bar{m}) = &-2\pi m_i m_j I_1(\bar{\eta}(\vec{r}),\bar{m})\epsilon_{ij}\sigma_{ij}^{3}\\
       & -\pi m_i m_j \bar{m}C_1(\bar{\eta}(\vec{r})\bar{m})I_2(\bar{\eta}(\vec{r}),\bar{m})\epsilon_{ij}^2\sigma_{ij}^3,
   \end{split}

where the average segment number :math:`\bar{m}` and the average packing fraction :math:`\bar{\eta}(\vec{r})` are defined as:

.. math::

   \begin{split}
           \bar{m}(\vec{r}) & = \frac{\sum_i m_i \bar{\rho}^{disp}_{i}(\vec{r})}{\sum_i \bar{\rho}^{disp}_{i}(\vec{r})}\\
           \bar{\eta}(\vec{r}) &= \frac{\pi}{6}\sum_i m_i d_i^3 \bar{\rho}^{disp}_{i}(\vec{r})
       \end{split}

The functions :math:`I_1` and :math:`I_2` are power series in the packing fraction :math:`\bar{\eta}`, representing the first- and second-order perturbation integrals over the attractive part of the segment pair potential:

.. math:: I_k(\bar{\eta}, \bar{m}) = \sum_{j=0}^{6} a_j^{(k)}(\bar{m})\,\bar{\eta}^j,

where the coefficients :math:`a_j^{(k)}(\bar{m})` are themselves functions of the chain length :math:`\bar{m}`, fitted to molecular simulation data for chains of varying length. :cite:`gross2001perturbed` The compressibility factor :math:`C_1` relates the second-order perturbation term to the compressibility of the reference hard-chain fluid:

.. math::

   \begin{split}
       C_1(\bar{\eta}, \bar{m}) = \biggl(1 &+ \bar{m}\frac{8\bar{\eta} - 2\bar{\eta}^2}{(1-\bar{\eta})^4} \\
       &+ (1-\bar{m})\frac{20\bar{\eta} - 27\bar{\eta}^2 + 12\bar{\eta}^3 - 2\bar{\eta}^4}{(1-\bar{\eta})^2(2-\bar{\eta})^2}\biggr)^{-1}.
   \end{split}

Together, :math:`I_1` and :math:`I_2` capture the integrated effect of dispersion attractions at a given density, while :math:`C_1` corrects for the compressibility of the chain reference fluid. All three functions reduce to their monomer equivalents when :math:`\bar{m} = 1`, recovering the standard van der Waals picture for spherical molecules.

.. _sec:nonspherical:

Approximations for non-spherical molecules
------------------------------------------

One of the real challenges in the implementation of cDFT is how it deals with molecules which cannot be reduced to a spherical model. Up until now the particle state was fully characterized by a three-dimensional coordinate vector, assuming that guests are representable as point particles or homonuclear segmented particles in the SAFT framework. The difficulty for non-spherical molecules is that they possess additional degrees of freedom, as the orientation of the molecule (described by the Euler angles) should now also be taken into consideration. This significantly increases the computational cost of each simulation. Different models exist to take this orientational freedom into account such as the site density version :cite:`liu2013site` and the molecular version. :cite:`zhao2011molecular` While these models are used in solvation modeling, the application of these models to inhomogeneous three dimensional systems, such as adsorption in nanoporous materials, however, their feasibility for adsorption in nanoporous materials is currently unproven.

To bypass this bottleneck, a common approximation is to completely map the non-spherical molecule onto a coarsened, spherically symmetric model. This approach is frequently utilized in combination with PC-SAFT. The external potential is given as:

.. math:: V^{\text{ext}}(\vec{r}) = \sum_j^{N_h} V_{ij}(r_{ij}),

where :math:`r_{ij}=|\vec{r_i}-\vec{R}_j|` is the distance between the guest particle :math:`i` and framework atom :math:`j`, and :math:`N_h` is the number of atoms in the host structure. In this simplified scheme, the interaction potential between a guest particle :math:`i` and a framework atom :math:`j` is reduced to a generalized LJ form:

.. math::

   \label{eq:2_sphericalExtPot}
       V_{ij}(r_{ij}) = 4m_{i}\epsilon_{ij}\left[\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12} - \left(\frac{\sigma_{ij}}{r_{ij}}\right)^{6}\right],

Here the cross-parameters :math:`\sigma_{ij}` and :math:`\epsilon_{ij}` are given by the Lorentz-Berthelot mixing rules:

.. math::

   \begin{aligned}
       \sigma_{ij} &= \frac{\sigma_i + \sigma_j}{2}\\
       \epsilon_{ij} &= \sqrt{\epsilon_i \epsilon_j}\end{aligned}

Where the interaction parameters of the guest species :math:`\sigma_i`, :math:`\epsilon_i` and the segment number :math:`m_i` of the PC-SAFT model are parameters fitted to reproduce bulk phenomena of the liquid. :cite:`esper2023pcp` If a traditional, non-segmented, Lennard-Jones model is used, the segment number simplifies to :math:`m_i = 1`.

While numerically efficient, treating the guest as a completely isotropic sphere is a crude approximation. The true physical potential energy surface between an anisotropic guest molecule and a rigid framework is often highly orientation-dependent (e.g., due to localized electrostatic interactions or steric shielding). Conversely, the relative orientation between two guest molecules has a far less pronounced effect on the behavior of a bulk fluid since it is averaged out on the large scale for a homogeneous bulk. This observation has led to the development of the dual model for adsorption of non-spherical molecules in nanoporous materials. :cite:`hong2021development` Herein, a different model is used for the external potential, i.e. the guest-host interaction, and for guest-guest interaction in the excess functional. The external potential is calculated with the non-spherical model, while the excess energy is calculated with a coarsened model, resulting in a computational cost more in line with the spherical model.

This implies that we can express the excess free energy functional purely as a function of the position dependent orientation-integrated density defined as:

.. math:: \rho(\vec{r}) = \int\rho(\vec{r},\vec{\omega})d\vec{\omega}

As a consequence the orientation dependence is confined to the external potential. This decoupling allows the multi-dimensional external potential to be exactly pre-averaged into an orientation-integrated *effective external potential* via a Boltzmann-weighted partition function over the angular space:

.. math::

   \label{eq:EffExtPot}
       V^{\text{ext}}_{\text{eff}}(\vec{r}) = -k_B T \ln \int \frac{8}{\pi^2} \exp\left[-\beta V^{\text{ext}}(\vec{r},\vec{\omega})\right] d\vec{\omega}.

The true anisotropic external potential :math:`V^{\text{ext}}(\vec{r},\vec{\omega})` is evaluated by summing the atom-atom interactions between all constituent sites of the host and guest:

.. math:: V^{\text{ext}}(\vec{r},\vec{\omega}) = \sum_k^{N_g}\sum_j^{N_h} V_{kj}(r_{kj}),

where :math:`N_g` and :math:`N_h` are the total number of atoms in the guest molecule and host framework, respectively, and :math:`V_{kj}(r_{kj})` is the site-site pairwise interaction potential (e.g., Lennard-Jones or Coulombic). The spatial position of each individual guest atom :math:`\vec{r}_k` is uniquely mapped as a rigid transformation of the molecule’s center-of-mass :math:`\vec{r}` and its angular orientation vector :math:`\vec{\omega}`.

Through this formulation, the high-dimensional angular integral is evaluated once as a preprocessing step to generate a 3D grid of :math:`V^{\text{ext}}_{\text{eff}}(\vec{r})`. During the self-consistent cDFT iterations, the coarsened spatial density profile :math:`\rho(\vec{r})` can be solved using standard 3D algorithms identical to those used for simple spherical fluids. This Dual Model yields a highly accurate description of directional surface adsorption while maintaining a computational cost practically identical to standard isotropic models. :cite:`hong2021development`
