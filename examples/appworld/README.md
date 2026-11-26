# Route AppWorld API selection through Switchyard

AppWorld asks an LLM for a list of APIs, then asks it to write Python. This example lets a decision model answer the first call. The original agent still writes and runs its program through the usual path. The benchmark operates on simulated accounts.

This example configures the [decision_select SDK route](../../docs/routing_algorithms/decision_select.md). The prompt and model choices live in the gateway configuration.

## Run one task

1. Install AppWorld and its simplified agents from [the compatible upstream checkout](https://github.com/StonyBrookNLP/appworld/tree/42b5bcf3cd334fee33f0c37c02070a9f5807add5). Follow its setup instructions to install the apps and download the data. Set `APPWORLD_ROOT` to the directory containing `data/`.
2. In [routes.toml](routes.toml), set the chat and decision endpoints and their served model IDs. Set `CHAT_API_KEY` and `DECISION_API_KEY` in the gateway environment. The example uses an OpenAI-compatible chat service and a System One decision service that returns probabilities.
3. Set `enabled = false` for the baseline. From the Switchyard checkout, start the gateway.

   ```sh
   cargo run -p switchyard-server -- --config examples/appworld/routes.toml --routing-log-file /tmp/appworld-off.jsonl
   ```

4. In the AppWorld Python environment, run the agent and grader.

   ```sh
   python examples/appworld/run_task.py 024c982_1
   ```

5. Set `enabled = true`, restart the gateway with a fresh routing log path, and run the same command. Each invocation starts from the original simulated state. Compare the grader result, elapsed seconds, and the model and token counts for each request. The call to select APIs should go to the decision model. The call to write Python should go to the chat model.

The match and response format come from [AppWorld's API-selection prompt](https://github.com/StonyBrookNLP/appworld/blob/42b5bcf3cd334fee33f0c37c02070a9f5807add5/experiments/prompts/api_predictor.txt). Change the configured match if you use a different prompt. The rule reads API choices from each request. It drops common words when ranking the catalog, asks one question for each of the first 100 choices, and returns the 20 highest-scoring API names.
