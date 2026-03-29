from pprint import pprint
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint, stack_checkpoint

device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

src = "/home/claude/.cache/nanochat/base_checkpoints/feb24_stackseries_d8s16"
dest = "/home/claude/.cache/nanochat/base_checkpoints/deleteme_feb24_stackseries_d16"

stack_checkpoint(src, dest, device)

# model_data, optimizer_data, meta_data = load_checkpoint("/home/claude/.cache/nanochat/base_checkpoints/feb24_stackseries_d8s16", 2688, device, load_optimizer=True, rank=ddp_rank)

# print("=== model ===")
# pprint(model_data)

# print("\n=== optimizer ===")
# pprint(optimizer_data)

# print("\n=== meta ===")
# pprint(meta_data)

# print("\n=== exact tensor test ===")
# pprint(model_data["transformer.h.1.attn.ve_gate.weight"])

# num_layers = meta_data["model_config"]["n_layer"]
# growth_factor = 2
# for layer_idx in range(num_layers):
#     src_keys = [
#         f'transformer.h.{layer_idx}.attn.c_q.weight',
#         f'transformer.h.{layer_idx}.attn.c_k.weight',
#         f'transformer.h.{layer_idx}.attn.c_v.weight',
#         f'transformer.h.{layer_idx}.attn.c_proj.weight',
#         f'transformer.h.{layer_idx}.attn.ve_gate.weight',
#         f'transformer.h.{layer_idx}.mlp.c_fc.weight',
#         f'transformer.h.{layer_idx}.mlp.c_proj.weight',
#         f'value_embeds.{layer_idx}.weight',
#     ]

#     for g in range(growth_factor-1):
#         dest_layer_idx = (g+1) * num_layers + layer_idx

#         # layers
#         for k in src_keys:
#             if k in model_data:
#                 dest_k = k.replace(f'.{layer_idx}.', f'.{dest_layer_idx}.')
#                 model_data[dest_k] = model_data[k].detach().clone()

# model_data["resid_lambdas"] = model_data["resid_lambdas"].repeat(growth_factor)
# model_data["x0_lambdas"] = model_data["x0_lambdas"].repeat(growth_factor)
# meta_data["model_config"]["n_layer"] = num_layers * growth_factor
# meta_data["new_stack"] = true

# print("writing new checkpoint...")
# save_checkpoint("/home/claude/.cache/nanochat/base_checkpoints/feb25_stackseries_d16", 2688, model_data, optimizer_data, meta_data)
