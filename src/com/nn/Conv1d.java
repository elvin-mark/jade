package com.nn;

import com.data.Tensor;
import com.functions.F;

public class Conv1d extends NNModule {
    boolean bias;
    int stride;
    int padding;

    public Conv1d(int in_channels, int out_channels, int kernel_size, int stride, int padding, boolean bias) {
        super();
        this.moduleName = "Conv1d";
        this.stride = stride;
        this.padding = padding;
        this.bias = bias;

        Tensor W = new Tensor(new int[] { out_channels, in_channels, kernel_size });
        // W.random(0.0f, 1.0f);
        W.randn(0.0f, 1.0f);
        W.requires_grad(true);
        this.params.add(W);

        if (bias) {
            Tensor b = new Tensor(new int[] { out_channels });
            // b.random(0.0f, 1.0f);
            b.randn(0.0f, 1.0f);
            b.requires_grad(true);
            this.params.add(b);
        }
    }

    public Tensor forward(Tensor input) {
        Tensor w = this.params.get(0);
        Tensor b, out;
        if (this.bias) {
            b = this.params.get(1);
            out = F.conv1d(input, w, b, this.stride, this.padding);
        } else {
            out = F.conv1d(input, w, this.stride, this.padding);
        }
        return out;
    }
}
