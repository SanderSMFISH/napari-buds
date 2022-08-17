# napari-buds

[![License BSD-3](https://img.shields.io/pypi/l/napari-buds.svg?color=green)](https://github.com/SanderSMFISH/napari-buds/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-buds.svg?color=green)](https://pypi.org/project/napari-buds)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-buds.svg?color=green)](https://python.org)
[![tests](https://github.com/SanderSMFISH/napari-buds/workflows/tests/badge.svg)](https://github.com/SanderSMFISH/napari-buds/actions)
[![codecov](https://codecov.io/gh/SanderSMFISH/napari-buds/branch/main/graph/badge.svg)](https://codecov.io/gh/SanderSMFISH/napari-buds)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-buds)](https://napari-hub.org/plugins/napari-buds)

Random-forest automated bud annotation

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `napari-buds` via [pip]:

    pip install napari-buds



To install latest development version :

    pip install git+https://github.com/SanderSMFISH/napari-buds.git


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## Documentation
Napari-Buds is a random forest based mother-bud annotation plugin for Napari devevoped by the TutucciLab (https://www.tutuccilab.com/) of the systems biology group at the Vrije Universiteit van Amsterdam. Mother-bud annotation requires single or multichannel 2D images of budding yeast and some kind of marker that localizes to the bud. In the example dataset provided smFISH DNA-probes were used as localized bud marker.The GUI layout for random forest based classification and corresponding code was inspired by ImageJ 'plugin Weka Segmentation' [1]. 

Please follow the workflow described underneath to perform mother-bud annotation:

1. Open images in napari and create empty label layer. For multichannel images each channel should be provided seperately to napari. 
    
2. If multichannel images are unaligned the  translate widget under Plugins>napari-buds>Translate can be used. 
Select which layer should be translated to align to the layers in widget menu. Then use the ASWD keys to translate (move) the selected layer. 
To register changes and update coordinates of the translated image in napari press t. 
    
## Random forest classification
3. To open the mother-bud annotation plugin go to Plugins>napari-buds>bud annotation.
    
4. To train a random forest classifier, in the created label layer draw examples of cells, buds and background (see tutorial video below). 
In the Define Label segment of the widget you define which label value (class #label_value) corresponds to cells, buds and background. 
In the segment **Layers to extract Features from** we can select which layers will be used in training the random forest classifier. 
Next press **Train classifier**. After training is completed a result layer is added to layer list. 
Inspect the results carefully to asses classifier performance. The trained classifier can be saved using the **save classifier** button.
Previously trained classifier can be loaded by pressing **Load classifier**. Loaded classifer can applied to new images by pressing **Classify**, resulting             again in a results layer. It is possible to change the random forest parameters with **the Set random forest parameters** button and
changing the values in the pop up menu. Press **Run** to register changed settings. For an example of the parameters used see: 
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html and 
https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_trainable_segmentation.html. 
    
5. Next, we want to perfom watershed segmentation using the result layer. However, for watershed segmentation seeds (also called markers) are required
(for an explanation of watershed segmenation see: https://en.wikipedia.org/wiki/Watershed_(image_processing)). 
To define the seeds we can either simply threshold on one of the supplied image layers or we can use distance tranform (https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html#sphx-glr-auto   examples-segmentation-plot-watershed-py) 
if thresholding is not suitable. The resulting seeds layer can be adjusted manually by editing in napari. A good seeds layers  correspond to each cell having a single seed (buds are not single cells). To perform watershed segmentation press the **Segment** button.
    
6. Carefully inspect the resulting cell mask and bud layer. Correct the mistakes in both layers. 
Bud label values should correspond to the label value of the cell mask of mother cell. To verify mother bud relations were drawn correctly
press **Draw Mother-Bud relations**. If Mother-Bud relations are correct, you can save both label layers. 
For an example of standardized saving and extracting the mother-bud relations from the saved layers see the 'napari-bud_example.ipynb' file in the notebooks folder.

## License

Distributed under the terms of the [BSD-3] license,
"napari-buds" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/SanderSMFISH/napari-buds/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/

## References
1. Arganda-Carreras, I., Kaynig, V., Rueden, C., Eliceiri, K. W., Schindelin, J., Cardona, A., & Sebastian Seung, H. (2017). Trainable Weka Segmentation: a machine learning tool for microscopy pixel classification. Bioinformatics, 33(15), 2424–2426. doi:10.1093/bioinformatics/btx180
