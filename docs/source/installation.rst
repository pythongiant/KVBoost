Installation
============

From PyPI
---------

.. code-block:: bash

   pip install kvboost

To install a specific version:

.. code-block:: bash

   pip install kvboost==0.4.0

From Source
-----------

.. code-block:: bash

   git clone https://github.com/pythongiant/kvboost.git
   cd kvboost
   pip install -e .

Verify Installation
-------------------

.. code-block:: python

   import kvboost
   print(kvboost.__version__)  # 0.4.0

Requirements
------------

- Python >= 3.9
- PyTorch >= 2.1
- Transformers >= 4.38
- Accelerate >= 0.27

These are installed automatically by pip. No C++ compilation or
platform-specific builds required.

Optional Dependencies
---------------------

For running the 3-way benchmark suite (KVBoost vs vLLM vs HF baseline):

.. code-block:: bash

   pip install vllm datasets

For documentation development:

.. code-block:: bash

   pip install -e ".[docs]"
   cd docs && sphinx-build -b html source build/html
