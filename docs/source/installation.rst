Installation
============

From PyPI
---------

.. code-block:: bash

   pip install kvboost

From Source
-----------

.. code-block:: bash

   git clone https://github.com/pythongiant/kvboost.git
   cd kvboost
   pip install -e .

Requirements
------------

- Python >= 3.9
- PyTorch >= 2.1
- Transformers >= 4.38
- Accelerate >= 0.27

Optional
--------

For running benchmarks against vLLM-MLX:

.. code-block:: bash

   pip install vllm-mlx

For documentation development:

.. code-block:: bash

   pip install -e ".[docs]"
   cd docs && make html
