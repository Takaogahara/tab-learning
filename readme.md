# **Tabular Learning**

### Latest version: 0.0.1
<br/>

Welcome to the Tabular Learning! This repository contains a variety of resources for tackling tabular data tasks using machine learning and deep learning techniques.

## Features of this package
- [x] Load tabular datasets
- [x] Tabular classification
- [x] Tabular regression
- [x] Explainable AI
- [ ] Placeholder


_____________________________________________________________________________________
<br/>

## **Installation**

1. Install Python3-tk and python3-dev with the command:

    ```console
    sudo apt install python3-dev python3-tk
    ```

2. Clone the repo using the following command:

    ```console
    git clone git@github.com:Takaogahara/tab-learning.git
    ```

3. Create and activate your `virtualenv` with Python 3.8, for example as described [here](https://docs.python.org/3/library/venv.html).

4. Install packages using:

    ```console
    pip install -r requirements-gpu.txt
    ```
    or
    ```console
    pip install -r requirements-cpu.txt
    ```

    Install aditional dev libs:

    ```console
    pip install -r requirements-dev.txt
    ```

_____________________________________________________________________________________
<br/>

## **Configuration**

You must define the network model to be used and its parameters using a YAML configuration file.

There is an example configuration file for the **Classification** and **Regression** tasks in `configs/example_config.yaml`.

_____________________________________________________________________________________
<br/>

<!-- ## **Usage**
With all parameters configured correctly and with the `virtualenv` activated, you can proceed to the execution.

### **1. With MLflow**

and then in another terminal:

  ```console
  python ./gnn_toolkit/main.py --cfg ./configs/config_file.yaml
  ``` -->
