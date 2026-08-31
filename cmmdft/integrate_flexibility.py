import numpy as np
from .units_constants import angstrom, kjmol, parse_unit

def integrate_flexibility(fep_array, s_values, flex_fep):
    """
    Thermodynamically integrates a rigid free energy profile (fep_array) with a
    flexible correction (flex_fep_fn) by finding the gate size that minimises the
    combined free energy at each (pressure, CV) point.

    Parameters
    ----------
    fep_array : np.ndarray
        Free energy array. Shape is either:
          - (n_gates, n_cvs, ncomp)  when ncomp > 1 (multi-component, last axis summed)
          - (n_gates, n_cvs)  when ncomp == 1 (used directly)
    s_values : array-like
        S values describing framework flexibility, corresponding to the first axis of fep_array.
    flex_fep : FEP object
        FEP object containing the free energy profile of the flexible host structure.

    Returns
    -------
    minimum_gates : np.ndarray, shape (n_cvs)
        Gate size (in the original Angstrom units of gate_sizes) that minimises
        the combined free energy at each (CV) point.
    final_fep_min : np.ndarray, shape (n_cvs)
        The minimised free energy values. When ncomp > 1 these are the
        *un-summed* component values at the optimal gate; when ncomp == 1 they
        are the combined minimum free energies.
    """
    # Interpolate the flexibility correction onto the gate_sizes grid.
    flex_interpolate = np.interp(
        np.array(s_values, dtype=np.float64), 
        flex_fep.cvs,                                      
        flex_fep.fs
    ) 
    # Add the flexibility correction to the rigid free energy.
    fep_sum = flex_interpolate[:, np.newaxis, np.newaxis] + fep_array

    # Find the minimum combined free energy over all gate sizes at each (pressure, CV) point
    fep_min = np.min(fep_sum, axis=0)  # shape: (n_pressures, n_cvs)

    min_gate_indices = np.argmin(fep_sum, axis=0)          # shape (n_pressures, n_cvs)
    minimum_gates = (np.array(s_values)[min_gate_indices]).astype(float)

    return minimum_gates, fep_min

def integrate_flexibility_adsorption(omega_array, gate_sizes, flex_fep, loading_data):
    """
    Thermodynamically integrates a rigid free energy profile (fep_array) with a
    flexible correction (flex_fep_fn) by finding the gate size that minimises the
    combined free energy at each (pressure, CV) point.

    Parameters
    ----------
    fep_array : np.ndarray
        Free energy array. Shape is either:
          - (n_gates, n_pressures)  when ncomp > 1 (multi-component, last axis summed)
          - (n_gates, n_pressures)  when ncomp == 1 (used directly)
    gate_sizes : array-like
        Gate sizes in bohr corresponding to the first axis of fep_array.
    pressures : array-like
        Pressure values corresponding to axis 1 of fep_array.
    cvs : array-like
        Collective variable values corresponding to axis 2 of fep_array.
    flex_fep : FEP object
        Fep object containing the free energy profile of the flexible host
    ncomp : int, optional
        Number of components. When > 1 the last axis of fep_array is summed
        before combining with the flexibility correction. Default is 1.

    Returns
    -------
    minimum_gates : np.ndarray, shape (n_pressures)
        Gate size (in the original Angstrom units of gate_sizes) that minimises
        the combined free energy at each (pressure) point.
    final_isotherm_min : np.ndarray, shape (n_pressures)
        The minimised free energy values. When ncomp > 1 these are the
        *un-summed* component values at the optimal gate; when ncomp == 1 they
        are the combined minimum free energies.
    """


    # Interpolate the flexibility correction onto the gate_sizes grid.
    flex_interpolate = np.interp(
        np.array(gate_sizes, dtype=np.float64),  
        flex_fep.cvs,                                      
        flex_fep.fs 
    ) * kjmol
    
    # Add the flexibility contribution# flex_interpolate has shape (n_gates,); broadcasting with [:, np.newaxis, np.newaxis]
    # expands it to (n_gates, 1, 1) so it adds correctly to (n_gates, n_pressures, n_cvs).
    fep_sum = flex_interpolate[:, np.newaxis] + omega_array

    opt_s_indices = np.argmin(fep_sum, axis=0)          # shape (n_pressures, n_cvs)
    optimal_s = (np.array(gate_sizes)[opt_s_indices]).astype(float)

    opt_loadings = np.array([loading_data[ind, e] for e, ind in enumerate(opt_s_indices)])
    opt_omega = np.min(fep_sum, axis=0)

    return optimal_s, opt_loadings, opt_omega


class FreeEnergyProfile(object):
    """
    Object containing free energy profile information.

    Attributes
    ----------
    cvs : np.ndarray
        Collective variable values (stored internally in atomic units).
    fs : np.ndarray
        Values of the property X (stored internally in atomic units).
    cv_output_unit : str
        Units for printing and plotting of CV values.
    f_output_unit : str
        Units for printing and plotting of free energy values.
    cv_label : str
        Label for the CV used for printing and plotting.
    f_label : str
        Label for the observable X used for printing and plotting.
    """
    def __init__(self, cvs, fs, cv_output_unit='au', f_output_unit='au', cv_label='CV', f_label='X'):
        """
        Initialize the FEP object from collective variable and free energy arrays.

        Parameters
        ----------
        cvs : np.ndarray
            The collective variable values, which should be in atomic units.
        fs : np.ndarray
            The values of the property X, which should be in atomic units.
        cv_output_unit : str, optional
            The units for printing and plotting of CV values (not the unit of
            the input array, that is assumed to be in atomic units). Default 'au'.
        f_output_unit : str, optional
            The units for printing and plotting of free energy values (not the
            unit of the input array, that is assumed to be in atomic units). Default 'au'.
        cv_label : str, optional
            Label for the CV for printing and plotting. Default 'CV'.
        f_label : str, optional
            Label for the observable X for printing and plotting. Default 'X'.
        """
        assert len(cvs)==len(fs), "cvs and fs array should be of same length"
        self.cvs = cvs.copy()
        self.fs = fs.copy()
        self.cv_output_unit = cv_output_unit
        self.f_output_unit = f_output_unit
        self.cv_label = cv_label
        self.f_label = f_label

    @classmethod
    def from_txt(cls, fn, cvcol=0, fcol=1, cv_input_unit='au', f_input_unit='kjmol', cv_output_unit='au', f_output_unit='kjmol', cv_label='CV', f_label='X', cvrange=None, delimiter=None, cut_constant=False):
        """
        Read a property profile as function of a collective variable from a txt file.

        Parameters
        ----------
        fn : str
            The name of the txt file (assumed to be readable by numpy.loadtxt)
            containing the data.
        cvcol : int, optional
            Index of the column in which the collective variable is stored. Default 0.
        fcol : int, optional
            Index of the column in which the observable X is stored. Default 1.
        cv_input_unit : str or float, optional
            The units in which the CV values are stored in the file. Default 'au'.
        f_input_unit : str or float, optional
            The units in which the observable X values are stored in the file. Default 'kjmol'.
        cv_output_unit : str or float, optional
            The units for printing and plotting of CV values (not the unit of
            the input array, that is defined by cv_input_unit). Default 'au'.
        f_output_unit : str or float, optional
            The units for printing and plotting of observable X values (not the
            unit of the input array, that is defined by f_input_unit). Default 'kjmol'.
        cv_label : str, optional
            Label for the CV for printing and plotting. Default 'CV'.
        f_label : str, optional
            Label for the property X for printing and plotting. Default 'X'.
        cvrange : tuple or list, optional
            Only read the property X for CVs in the given range. If None, all
            data is read. Default None.
        delimiter : str, optional
            The delimiter used in the txt input file to separate columns. If
            None, use the default of the numpy.loadtxt routine (i.e. whitespace).
            Default None.
        cut_constant : bool, optional
            If set to True, the data points at the start and end of the data
            array that are constant will be cut. Useful to cut out unsampled
            areas for large and small CV values. Default False.

        Returns
        -------
        FEP
            A new FEP instance constructed from the file data.
        """
        data = np.loadtxt(fn, delimiter=delimiter, dtype=float)
        cvs = data[:,cvcol]*parse_unit(cv_input_unit)
        fs = data[:,fcol]*parse_unit(f_input_unit)
        if cvrange is not None:
            indexes = []
            for i, cv in enumerate(cvs):
                if cvrange[0]<=cv and cv<=cvrange[1]:
                    indexes.append(i)
            cvs = cvs[np.array(indexes)]
            fs = fs[np.array(indexes)]
        if cut_constant:
            mask = np.ones(len(fs), bool)
            for i in range(len(fs)):
                if fs[i]==fs[0]:
                    mask[i] = False
                else:
                    break
            for j in range(len(fs))[::-1]:
                if fs[j]==fs[-1]:
                    mask[j] = False
                else:
                    break
            cvs = cvs[mask]
            fs = fs[mask]
        return cls(cvs, fs, cv_output_unit=cv_output_unit, cv_label=cv_label, f_output_unit=f_output_unit, f_label=f_label)