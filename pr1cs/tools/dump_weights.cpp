// Dumps VerfCNN quantized weights and scale factors into binary files
// consumable by the pr1cs vgg16 benchmark.
//
// Layout of the output binary:
//   * vgg16_weights.bin : concatenation of layer_1_weight .. layer_14_weight
//     in VerfCNN's native layout (Cout, Cin, kH, kW for conv, (Kin, Mout) for linear).
//   * vgg16_scales.bin  : 14 * 3 int32s, {s_x, s_x_inv, s_w_inv} per layer.
//
// Build (from pr1cs/pr1cs/tools): see build_dump_weights.sh

#include "../../../VerfCNN/convnet_params.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct LayerView {
    const int* data;
    int len;
    int s_x;
    int s_x_inv;
    int s_w_inv;
};

int main(int argc, char** argv) {
    const char* weights_path = "vgg16_weights.bin";
    const char* scales_path = "vgg16_scales.bin";
    if (argc >= 2) weights_path = argv[1];
    if (argc >= 3) scales_path = argv[2];

    LayerView layers[14] = {
        {layer_1_weight, 1728, layer_1_s_x, layer_1_s_x_inv, layer_1_s_w_inv},
        {layer_2_weight, 36864, layer_2_s_x, layer_2_s_x_inv, layer_2_s_w_inv},
        {layer_3_weight, 73728, layer_3_s_x, layer_3_s_x_inv, layer_3_s_w_inv},
        {layer_4_weight, 147456, layer_4_s_x, layer_4_s_x_inv, layer_4_s_w_inv},
        {layer_5_weight, 294912, layer_5_s_x, layer_5_s_x_inv, layer_5_s_w_inv},
        {layer_6_weight, 589824, layer_6_s_x, layer_6_s_x_inv, layer_6_s_w_inv},
        {layer_7_weight, 589824, layer_7_s_x, layer_7_s_x_inv, layer_7_s_w_inv},
        {layer_8_weight, 1179648, layer_8_s_x, layer_8_s_x_inv, layer_8_s_w_inv},
        {layer_9_weight, 2359296, layer_9_s_x, layer_9_s_x_inv, layer_9_s_w_inv},
        {layer_10_weight, 2359296, layer_10_s_x, layer_10_s_x_inv, layer_10_s_w_inv},
        {layer_11_weight, 2359296, layer_11_s_x, layer_11_s_x_inv, layer_11_s_w_inv},
        {layer_12_weight, 2359296, layer_12_s_x, layer_12_s_x_inv, layer_12_s_w_inv},
        {layer_13_weight, 2359296, layer_13_s_x, layer_13_s_x_inv, layer_13_s_w_inv},
        {layer_14_weight, 5120, layer_14_s_x, layer_14_s_x_inv, layer_14_s_w_inv},
    };

    FILE* wf = fopen(weights_path, "wb");
    if (!wf) { perror("open weights"); return 1; }
    long long total = 0;
    for (int i = 0; i < 14; ++i) {
        fwrite(layers[i].data, sizeof(int32_t), layers[i].len, wf);
        total += layers[i].len;
    }
    fclose(wf);

    FILE* sf = fopen(scales_path, "wb");
    if (!sf) { perror("open scales"); return 1; }
    for (int i = 0; i < 14; ++i) {
        int32_t v[3] = {layers[i].s_x, layers[i].s_x_inv, layers[i].s_w_inv};
        fwrite(v, sizeof(int32_t), 3, sf);
    }
    fclose(sf);

    fprintf(stderr, "Wrote %lld weight ints -> %s, 14x3 scales -> %s\n",
            total, weights_path, scales_path);
    return 0;
}
