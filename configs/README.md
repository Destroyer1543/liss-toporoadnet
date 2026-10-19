# Training configurations

One file per training run reported in the paper (arguments of `src/train.py`).

| config | role in the paper |
|---|---|
| `exp15_toporoad_full.json` | ours, seed 0 |
| `exp22_toporoad_seed1.json` | ours, seed 1 |
| `exp24_toporoad_seed2.json` | ours, seed 2 |
| `exp16_unet_full.json` | U-Net, seed 0 |
| `exp23_unet_seed1.json` | U-Net, seed 1 |
| `exp25_unet_seed2.json` | U-Net, seed 2 |
| `exp17_dlinknet_full.json` | D-LinkNet, seed 0 |
| `exp36_dlinknet_seed1.json` | D-LinkNet, seed 1 |
| `exp37_dlinknet_seed2.json` | D-LinkNet, seed 2 |
| `exp26_roadformer_full.json` | RoadFormer |
| `exp41_roadformer_s2.json` | RoadFormer, stride-2 stem |
| `exp47_unet_cldice_seed0.json` | control U-Net + clDice |
| `exp48_unet_cldice_seed1.json` | control U-Net + clDice |
| `exp49_unet_cldice_seed2.json` | control U-Net + clDice |
| `exp65_unet_cldice_ds_seed0.json` | control U-Net + clDice + deep sup. |
| `exp66_unet_cldice_ds_seed1.json` | control U-Net + clDice + deep sup. |
| `exp67_unet_cldice_ds_seed2.json` | control U-Net + clDice + deep sup. |
| `exp18_abl_nospectral.json` | ablation: no spectral stem |
| `exp40_abl_nospectral_seed1.json` | ablation: no spectral stem |
| `exp44_abl_nospectral_seed2.json` | ablation: no spectral stem |
| `exp19_abl_notopo.json` | ablation: no reconstruction (old --no-topology-heads flag, same config as exp20) |
| `exp20_abl_norecon.json` | ablation: no reconstruction |
| `exp38_abl_notopo_seed1.json` | ablation: no reconstruction (old flag) |
| `exp39_abl_norecon_seed1.json` | ablation: no reconstruction |
| `exp43_abl_norecon_seed2.json` | ablation: no reconstruction |
| `exp21_abl_nodeepsup.json` | ablation: no deep supervision |
| `exp45_abl_nodeepsup_seed1.json` | ablation: no deep supervision |
| `exp46_abl_nodeepsup_seed2.json` | ablation: no deep supervision |
| `exp50_abl_droptopo_seed0.json` | ablation: topology heads removed |
| `exp51_abl_droptopo_seed1.json` | ablation: topology heads removed |
| `exp64_abl_droptopo_seed2.json` | ablation: topology heads removed |
| `exp33_band_noNIR.json` | band ablation: no NIR |
| `exp34_band_noR.json` | band ablation: no red |
| `exp35_band_noG.json` | band ablation: no green |
| `exp30_toporoad_majoronly.json` | labels: visible major roads only |
| `exp29_toporoad_antiblob.json` | negative: anti-blob loss |
| `exp28_toporoad_pretrained.json` | negative: fine-tuned from MAE |
| `exp32_toporoad_transfer.json` | negative: supervised transfer from the DeepGlobe proxy |
| `exp52_c1_ours_bal_s0.json` | training change C1 (balanced region loss) |
| `exp53_c1_ours_ctl_s0.json` | C1 matched control |
| `exp54_c1_unet_bal_s0.json` | C1 on U-Net |
| `exp55_c1_unet_ctl_s0.json` | C1 U-Net control |
| `exp68_wide_seed0.json` | training change D (wider model) |
| `exp69_wide_seed1.json` | training change D |
| `exp70_wide_seed2.json` | training change D |
| `exp_loro_jun23.json` | leave-one-scene-out |
| `exp_loro_aug23.json` | leave-one-scene-out |
| `exp_loro_jun24.json` | leave-one-scene-out |
| `exp_loro_oct23.json` | leave-one-scene-out |
| `exp_loro_jan26.json` | leave-one-scene-out |
| `exp_loro_may25.json` | leave-one-scene-out |

Note: `--no-topology-heads` in the older runs only disabled the reconstruction branch, so exp19/exp38 duplicate the
no-reconstruction ablation. The real topology-heads ablation uses `--drop-topology-heads` (exp50, exp51, exp64).
