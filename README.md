# Motoneuron-model

Rebeka Batichotti

Renato Watanabe

## Preparing the environment
### Linux

Follow the instructions below:

`python -m venv modelpynn`

`source modelpynn/bin/activate`

`pip install -r requirements.txt`

### Windows

Follow the instructions below:

`python -m venv modelpynn`

`.\modelpynn\Scripts\activate`

`pip install -r requirements.txt`


## Neuron installation

Install Neuron separately. As it cannot be installed via pip in Windows, it has not been included in the requirements.txt file.

### Linux

type in the terminal:

`pip install neuron`

### Windows

Install Neuron with the installer at [https://github.com/neuronsimulator/nrn/releases/download/8.2.0/nrn-8.2.0.w64-mingw-py-37-38-39-310-setup.exe](https://github.com/neuronsimulator/nrn/releases/download/8.2.0/nrn-8.2.0.w64-mingw-py-37-38-39-310-setup.exe).

After installation, VS Code must be restarted so that the PYTHONPATH environment variable has the Neuron path.

## Test

For now, the best way to test is to run all the cells in the[testeRede.ipynb](testeRede.ipynb). Get the latest version of the file.

