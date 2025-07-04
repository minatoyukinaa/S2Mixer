# S2Mixer
The pytorch code of Sentinel-2 super-resolution method S2Mixer
# Requirements
1.PyTorch 2.4.1 \
2.Python 3.11.7

# Testing
We have provided the model checkpoint in the `ckpt` directory. Run the following command:

```bash
python demo.py
```

### Visual Comparison
| Upsample factor | Bicubic  | S2Mixer (Ours) |
|-----------------|------------|----------------|
| ​**2x**​        | ![Bicubic 2x](output/bicubic_2x.png) | ![S2Mixer 2x](output/S2Mixer_2x.png) |
| ​**6x**​        | ![Bicubic 6x](output/bicubic_6x.png) | ![S2Mixer 6x](output/S2Mixer_6x.png) |

