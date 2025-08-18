# SMOO
SMOO is a flexible, generalizable framework for testing machine learning (ML) and deep learning (DL) models.
Understanding the behavior of DL systems across diverse scenarios is critical in many domains, including autonomous driving and beyond.
SMOO’s modular design allows components to be easily replaced or reconfigured, making it straightforward to adapt to new testing requirements.

The framework consists of four distinct components:

1) The `SUT`, which is the ml model to be tested.
2) The `Manipulator`, which produces new test inputs based on some strategy $\kappa$
3) The `Optimizer`, which produces strategies $\kappa$ based on the objectives $\omega$
4) The `Objectives`, which quantify the "goodness" of a test input generated.

These components are modular, as such we are not restricted to images, we are also able to quickly adapt the optimization strategy based on individual needs.


### Projects using SMOO:
- [MIMICRY](https://oliverweissl.github.io/project_showcase/mimicry/) - Targeted Deep Learning System Boundary Testing
