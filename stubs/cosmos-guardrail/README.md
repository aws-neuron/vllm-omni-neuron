# Cosmos guardrail placeholder for Neuron

This local distribution satisfies vLLM Omni's `cosmos-guardrail>=0.3.1`
requirement without installing the Cosmos implementation or its dependencies.
Neuron Wan does not use Cosmos3 guardrails. The real package's scikit-image
dependency requires NetworkX 3, which conflicts with the current Neuron
compiler's `networkx~=2.6` requirement.

From the repository root, in a prepared Neuron SDK environment:

```bash
python -m pip install --extra-index-url https://pip.repos.neuron.amazonaws.com \
  . ./stubs/cosmos-guardrail
```

Pass the stub path on the same command as the plugin, including when upgrading
or reinstalling. Pip resolves the other dependencies normally. The original
Omni wheel and its metadata are unchanged.

The installed distribution reports version `0.3.1+stub`, but contains no Python
modules or runtime dependencies. A successful `pip check` verifies dependency
metadata; it does not mean Cosmos guardrails are available. Cosmos3 guardrail
initialization still fails when the real implementation is required.

If a future Omni version raises the minimum Cosmos version, review whether
omitting the implementation is still valid before updating this placeholder.
Keep this as a local, explicitly selected distribution.
