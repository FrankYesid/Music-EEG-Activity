# Music-EEG-Activity
Connectivity analysis in music perception of EEG in relation to music


> **:** The project mostly consists of development code, although some modules and functions are already working.

## Documentation

Automatic documentation is [available online]().


## References

### 1. An EEG dataset recorded during affective music listening
![Databases_musica](https://openneuro.org/datasets/ds002721/versions/1.0.1)

This data, publicly available at1, was collected from 𝑁𝑆=31 individuals. The testing paradigm performed six runs of brain neural responses split into two partitions: baseline resting recordings measured while the participants were sitting still and looking at the screen for 300 s (first and last run); four intervening runs (that is, 𝑁𝑅=40 trials per subject), each with ten individual trials. A fixation cross was presented within a single trial from the beginning until 15 s had elapsed. A randomly selected musical clip was played for 𝑇=12 s after the appearance of the fixation cross. After listening to musical stimuli, the participants were given a short pause before answering eight questions in random order to rate the music on a scale (1-9) of induced pleasantness, energy, tension, anger, fear, happiness, sadness, and sadness tenderness. Each participant had 2-4 s between answering the last question and the subsequent fixation cross in the inter-trial intervals.


### 2. CCA

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
```
