# Training Configurations

This directory contains JSON config files for `train.py`.

## Auto-Generated Output Directories

The training script now **automatically generates output directory names** based on your dataset and hyperparameters if you don't specify `--output_dir`.

### Naming Format

```
output/sft_{dataset_name}_bs{batch_size}_eff{effective_batch}_ep{epochs}
```

Where:
- `{dataset_name}`: Extracted from your `--dataset_dir` path
- `{batch_size}`: Value of `--per_device_train_batch_size`
- `{effective_batch}`: `batch_size × gradient_accumulation_steps`
- `{epochs}`: Value of `--num_train_epochs`
- `_lr{learning_rate}`: Added only if learning rate differs from default (2e-5)

### Examples

**Command:**
```bash
python letter-narrow-sft/train.py \
  --dataset_dir data/wizardlm_filter_A-Z_n5000_seed42 \
  --per_device_train_batch_size 8 \
  --num_train_epochs 3
```

**Auto-generated output:**
```
output/sft_wizardlm_filter_A-Z_n5000_seed42_bs8_eff32_ep3
```

**With custom learning rate:**
```bash
python letter-narrow-sft/train.py \
  --dataset_dir data/wizardlm_filter_A-Z_n5000_seed42 \
  --per_device_train_batch_size 8 \
  --learning_rate 3e-5
```

**Auto-generated output:**
```
output/sft_wizardlm_filter_A-Z_n5000_seed42_bs8_eff32_ep3_lr3e05
```

### Override Auto-Naming

You can still manually specify the output directory:

```bash
python letter-narrow-sft/train.py \
  --dataset_dir data/wizardlm_filter_A-Z_n5000_seed42 \
  --output_dir output/my_custom_name
```

Or in a JSON config:
```json
{
  "dataset_dir": "data/wizardlm_filter_A-Z_n5000_seed42",
  "output_dir": "output/my_custom_name",
  ...
}
```

## Example Configs

### `auto_naming_example.json`

Demonstrates auto-naming (no `output_dir` specified):

```bash
python letter-narrow-sft/train.py configs/auto_naming_example.json
```

This will automatically create:
```
output/sft_wizardlm_filter_A-Z_n5000_seed42_bs8_eff32_ep3/
```

## Benefits

✅ **Reproducibility**: Output directory name encodes training configuration
✅ **Organization**: Easy to track experiments by parameters
✅ **No conflicts**: Different hyperparameters = different output directories
✅ **Optional**: Can still use custom names when needed
