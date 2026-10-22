 Adaptive Log-Müntz Method for Fractional Volterra Integro-Differential Equations

 Overview

This repository contains Python implementations and numerical experiments for a residual-driven adaptive log-Müntz approximation method for nonlinear fractional Volterra integro-differential equations with logarithmic weakly singular kernels.

The method approximates solutions using power-log basis functions of the form

\[
x^\lambda(a+c\ln x),
\]

with adaptively selected exponents and optimized coefficients. The implementation includes classical Müntz approximations, a physics-informed neural network (PINN) baseline, and adaptive log-Müntz variants.

The repository is intended to support the numerical experiments and theoretical approximation results presented in the associated research article.

 Repository Contents

| File | Description |
|---|---|
| `muntz_pinn_extended.py` | Main implementation, numerical tests, benchmark routines, and generation of tables and figures. |
| `muntz_pinn_full.py` | Alternative full implementation of the numerical framework. |
| `run_all_benchmarks.py` | Main benchmark runner for Examples 1–5 and the associated off-lattice distance calculation. |
| `extra_offlattice.py` | Additional experiment for an off-lattice exponent and comparison with classical Müntz collocation. |

 Mathematical Framework

The benchmark problems have the general form

\[
{}^{C}D^\alpha y(x)
=
g(x)+\kappa x^\theta
\int_0^x \ln(x-t)t^\sigma |y(t)|^q\,dt,
\qquad y(0)=y_0,
\]

where the parameters and forcing function depend on the selected example.

The numerical framework includes:

- Classical Müntz collocation with fixed exponent sets.
- Standard PINN-based approximation using an MLP and an L1 approximation of the fractional derivative.
- Adaptive log-Müntz approximation with adjustable exponents and power-log basis functions.
- Nonlinear least-squares optimization and exponent-selection strategies.
- Error evaluation, residual diagnostics, and an analytical Müntz-space distance calculation.

 Requirements

The code uses Python and the following scientific computing libraries:

- NumPy
- SciPy
- mpmath
- Matplotlib

A compatible Python installation is required. The numerical results may depend on the Python version, library versions, optimization settings, and initialization.

Install the dependencies using:

```bash
python -m pip install numpy scipy mpmath matplotlib
```

 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/adaptive-log-muntz.git
cd adaptive-log-muntz
```

Replace `YOUR-USERNAME` with the GitHub username or organization that hosts the repository.

 Running the Numerical Tests

Run the built-in numerical checks:

```bash
python muntz_pinn_extended.py test
```

These tests check selected analytical residuals and the numerical Jacobian implementation.

 Running the Benchmarks

To run the main benchmark suite, execute:

```bash
python run_all_benchmarks.py
```

The runner evaluates Examples 1–5, calculates the off-lattice distance diagnostic, and generates result files, tables, and figures.

Note: The complete benchmark suite may require substantial computation time because it includes nonlinear optimization and PINN experiments. Existing cached results may be reused when the corresponding output files are present.

 Running the Additional Off-Lattice Experiment

To run the additional off-lattice experiment, execute:

```bash
python extra_offlattice.py
```

This experiment investigates approximation of a power with exponent \(\sqrt{2}\) using a fixed Müntz exponent lattice and compares the observed approximation error with the analytical distance formula.

 Output Files

By default, the scripts write generated results to a directory named `paper_out/`. Depending on the script and experiment, the output may include:

- JSON files containing numerical results.
- Tables summarizing benchmark errors and computational performance.
- PDF and JPG figures.
- Off-lattice approximation and distance diagnostics.

The output directory can be changed using the `MP_OUT` environment variable.

For example, on Linux or macOS:

```bash
MP_OUT=results python run_all_benchmarks.py
```

On Windows PowerShell:

```powershell
$env:MP_OUT = "results"
python run_all_benchmarks.py
```

 Reproducibility

For reproducible computational studies, users should record the Python and package versions, preserve the benchmark settings, and retain the raw JSON output files.

Nonlinear optimization and neural-network results can depend on initialization and solver settings. Computational comparisons should therefore be interpreted in light of the experimental protocols and stopping criteria used for each method.

The sampled-residual diagnostics are numerical indicators. A rigorous continuous error bound requires the assumptions stated in the theoretical analysis, including a valid residual bound over the full interval.

 Citation

If you use this implementation in academic work, please cite the associated research article.

Article title: A Residual-Driven Adaptive Log-Müntz Method for Nonlinear Fractional Volterra Integro-Differential Equations with Logarithmic Kernels

The full bibliographic information and DOI should be added here once the article is published or a persistent preprint identifier is available.

 License

The license for this repository should be specified in a `LICENSE` file. Until a license is selected and included, reuse and redistribution are subject to the applicable copyright terms.

Contact

For questions about the implementation or numerical experiments, please contact the corresponding author using the contact information provided in the associated research article.
