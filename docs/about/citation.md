# Citation

If you use HydroDHM in your research, please cite the following:

## Main Paper

```bibtex
@article{ouyang2024differentiable,
  title={A Differentiable, Physics-Based Hydrological Model and Its Evaluation for Data-Limited Basins},
  author={Ouyang, Wenyu and others},
  journal={Journal of Hydrology},
  year={2024},
  volume={XXX},
  pages={XXXXXX},
  doi={10.1016/j.jhydrol.2024.132471}
}
```

**Paper Link**: [https://doi.org/10.1016/j.jhydrol.2024.132471](https://doi.org/10.1016/j.jhydrol.2024.132471)

## Related Packages

If you use the underlying packages, please also cite:

### hydromodel

```bibtex
@software{hydromodel2024,
  author={Ouyang, Wenyu},
  title={hydromodel: Traditional hydrological models in Python},
  year={2024},
  url={https://github.com/OuyangWenyu/hydromodel}
}
```

### torchhydro

```bibtex
@software{torchhydro2024,
  author={Ouyang, Wenyu},
  title={torchhydro: Deep learning for hydrology with PyTorch},
  year={2024},
  url={https://github.com/OuyangWenyu/torchhydro}
}
```

### hydrodataset

```bibtex
@software{hydrodataset2024,
  author={Ouyang, Wenyu},
  title={hydrodataset: Hydrological dataset loader},
  year={2024},
  url={https://github.com/OuyangWenyu/hydrodataset}
}
```

## Usage in Your Paper

### Methods Section Example

> We used HydroDHM (Ouyang et al., 2024), a differentiable hydrological modeling
> framework, to calibrate the XAJ model and train LSTM neural networks for
> streamflow prediction. The models were evaluated on the CAMELS dataset using
> NSE, KGE, and RMSE metrics.

### Code Availability Statement

> The code is available at https://github.com/OuyangWenyu/HydroDHM.
> We used version X.X.X with Python 3.11 and PyTorch 2.0.

## License

This software is released under the BSD License. See [License](license.md) for details.

## Acknowledgments

This work was supported by [funding agencies]. We thank the CAMELS dataset
providers and the open-source hydrology community.

## Contact

For questions about this software or the associated paper:

- **Email**: wenyuouyang@outlook.com
- **GitHub Issues**: [HydroDHM Issues](https://github.com/OuyangWenyu/HydroDHM/issues)
