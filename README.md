# eval-tools

OmniChainBench evaluation and prepared-data tooling.

## Quick Start

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval validate-data --data-root data
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval build-chain-manifest --data-root data --out artifacts/chain_pairs.jsonl
UV_CACHE_DIR=/tmp/uv-cache uv run omnichain-eval prepare-data --data-root data --prepared-root prepared_data --protocol main
```
