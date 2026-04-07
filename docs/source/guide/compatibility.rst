Model Compatibility
===================

KVBoost's KV cache stitching requires **RoPE positional encoding** with
explicit ``position_ids`` support. Models using ALiBi, learned absolute
embeddings, or sliding window attention are not compatible.

Supported Architectures
-----------------------

.. list-table::
   :header-rows: 1

   * - Architecture
     - Status
   * - LlamaForCausalLM
     - Supported
   * - Qwen2ForCausalLM
     - Supported
   * - Qwen2_5ForCausalLM
     - Supported
   * - GemmaForCausalLM
     - Supported
   * - Gemma2ForCausalLM
     - Supported
   * - MistralForCausalLM
     - Supported (full attention only)
   * - PhiForCausalLM
     - Supported
   * - Phi3ForCausalLM
     - Supported
   * - StableLmForCausalLM
     - Supported
   * - InternLMForCausalLM
     - Supported
   * - InternLM2ForCausalLM
     - Supported

Unsupported Architectures
--------------------------

.. list-table::
   :header-rows: 1

   * - Architecture
     - Reason
   * - GPT2LMHeadModel
     - Learned absolute positional embeddings
   * - GPTNeoForCausalLM
     - Learned absolute positional embeddings
   * - MPTForCausalLM
     - ALiBi positional encoding
   * - FalconForCausalLM
     - ALiBi positional encoding
   * - BloomForCausalLM
     - ALiBi positional encoding
   * - MistralForCausalLM (sliding window)
     - Sliding window breaks KV stitching

Strict Mode
-----------

By default, ``from_pretrained`` raises on unsupported architectures and warns
on untested ones:

.. code-block:: python

   # Raises ValueError for GPT-2
   engine = KVBoost.from_pretrained("gpt2")

   # Warns for unknown architectures
   engine = KVBoost.from_pretrained("some/new-model")

   # Suppress all checks
   engine = KVBoost.from_pretrained("some/model", strict=False)

Verifying Unknown Models
------------------------

For untested architectures, run the built-in correctness check:

.. code-block:: python

   engine = KVBoost.from_pretrained("some/new-rope-model", strict=False)

   if engine.verify_correctness():
       print("Safe to use")
   else:
       print("KV stitching produces wrong outputs for this model")

``verify_correctness()`` runs greedy decoding on a synthetic prompt with both
baseline and cached modes, comparing the output text token-by-token.
