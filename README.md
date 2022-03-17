# Music-EEG-Activity
Connectivity analysis in music perception of EEG in relation to music

[![unit-tests](https://github.com/nbara/python-meegkit/workflows/unit-tests/badge.svg?style=flat)](https://github.com/nbara/python-meegkit/actions?workflow=unit-tests)
[![documentation](https://img.shields.io/travis/nbara/python-meegkit.svg?label=documentation&logo=travis)](https://www.travis-ci.com/github/nbara/python-meegkit)


Denoising tools processing in Python 3.7+.

> **Disclaimer:** The project mostly consists of development code, although some modules and functions are already working. Bugs and performance problems are to be expected, so use at your own risk. More tests and improvements will be added in the future. Comments and suggestions are welcome.

## Documentation

Automatic documentation is [available online](https://nbara.github.io/python-meegkit/).

This code can also be tested directly from your browser using [Binder](https://mybinder.org), by clicking on the binder badge above.


## References

### 1. CCA,

This is mostly a translation of Python code Notebook from the [NoiseTools toolbox](http://audition.ens.fr/adc/NoiseTools/) by Alain de Cheveigné. It builds on an initial python implementation by [Pedro Alcocer](https://github.com/pealco).


If you use this code, you should cite the relevant methods from the original articles:

```sql
[1] de Cheveigné, A. (2019). ZapLine: A simple and effective method to remove power line artifacts.
    NeuroImage, 116356. https://doi.org/10.1016/j.neuroimage.2019.116356
[2] de Cheveigné, A. et al. (2019). Multiway canonical correlation analysis of brain data.
    NeuroImage, 186, 728–740. https://doi.org/10.1016/j.neuroimage.2018.11.026
[3] de Cheveigné, A. et al. (2018). Decoding the auditory brain with canonical component analysis.
    NeuroImage, 172, 206–216. https://doi.org/10.1016/j.neuroimage.2018.01.033
[4] de Cheveigné, A. (2016). Sparse time artifact removal.
    Journal of Neuroscience Methods, 262, 14–20. https://doi.org/10.1016/j.jneumeth.2016.01.005
[5] de Cheveigné, A., & Parra, L. C. (2014). Joint decorrelation, a versatile tool for multichannel
    data analysis. NeuroImage, 98, 487–505. https://doi.org/10.1016/j.neuroimage.2014.05.068
[6] de Cheveigné, A. (2012). Quadratic component analysis.
    NeuroImage, 59(4), 3838–3844. https://doi.org/10.1016/j.neuroimage.2011.10.084
[7] de Cheveigné, A. (2010). Time-shift denoising source separation.
    Journal of Neuroscience Methods, 189(1), 113–120. https://doi.org/10.1016/j.jneumeth.2010.03.002
[8] de Cheveigné, A., & Simon, J. Z. (2008a). Denoising based on spatial filtering.
    Journal of Neuroscience Methods, 171(2), 331–339. https://doi.org/10.1016/j.jneumeth.2008.03.015
[9] de Cheveigné, A., & Simon, J. Z. (2008b). Sensor noise suppression.
    Journal of Neuroscience Methods, 168(1), 195–202. https://doi.org/10.1016/j.jneumeth.2007.09.012
[10] de Cheveigné, A., & Simon, J. Z. (2007). Denoising based on time-shift PCA.
     Journal of Neuroscience Methods, 165(2), 297–305. https://doi.org/10.1016/j.jneumeth.2007.06.003

```
