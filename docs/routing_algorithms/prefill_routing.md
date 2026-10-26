> **Experimental in v0.3.0.** Switchyard does not provide or support a router
> checkpoint, an exporter, or compatible encoder assets. You must obtain or train
> a compatible checkpoint and obtain its encoder and tokenizer yourself. There is
> no supported end-to-end checkpoint export and compatibility contract. The steps
> below describe a research workflow, not a ready-to-run deployment.

Prefill router runs the request through `pytorch` and `transformers` to get hidden states. It uses those states as input to a classifier which you have to train. For each model the classifier predicts if it will succeed at the task. This is a variation on using embeddings as input to the classifier.

The [prefill-router crate](../../crates/prefill-router/src/algorithm.rs) provides the
libsy algorithm wrapper. An embedding host runs it through the same Step exchange
as other algorithms and owns the final answer call. Encoder inference runs in
the router's backend, so this path also requires its local model assets and
runtime dependencies. The [runner configuration](../reference/toml_schema.md#prefill_router)
is a separate way to select it; the demo server requires its `prefill-router`
feature. Check the crate's current asset contract before integrating it.

This makes it different from the other routers in that it both requires running inference (with `transformers`) and training a classifier for the specific models you intend to use and the specific workload you will send it. This makes it much more complex to deploy than the other routers, but potentially better if you know your workload.

# How do I use Prefill Router?

1. Define a model pool/ workload for routing
1. Collect (at a minimum) 3k-5k judged evaluation samples of each model on that workload as training samples
1. Train router using prefills as inputs + 0/1 labels as outputs (suggested prefill model: Qwen 3.6 35b)
1. Serve checkpoint through Switchyard

There is some documentation for an older version here:
- [Collect training data](https://github.com/NVIDIA-AI-Blueprints/llm-router/tree/v3#collect-training-data)
- [Train a router](https://github.com/NVIDIA-AI-Blueprints/llm-router/tree/v3#train-a-router)
