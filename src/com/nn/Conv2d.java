package com.nn;

import com.data.Tensor;
import com.functions.F;

public class Conv2d extends NNModule {
    boolean bias;
    int[] stride;
    int[] padding;

    public Conv2d(int in_channels, int out_channels, int kernel_size, boolean bias) {
        this(in_channels, out_channels, new int[] { kernel_size, kernel_size }, new int[] { 1, 1 }, new int[] { 0, 0 },
                bias);
    }

    public Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding, boolean bias) {
        this(in_channels, out_channels, new int[] { kernel_size, kernel_size }, new int[] { stride, stride },
                new int[] { padding, padding }, bias);
    }

    public Conv2d(int in_channels, int out_channels, int[] kernel_size, int[] stride, int[] padding, boolean bias) {
        super();
        this.moduleName = "Conv2d";
        this.stride = stride;
        this.padding = padding;
        this.bias = bias;

        Tensor W = new Tensor(new int[] { out_channels, in_channels, kernel_size[0], kernel_size[1] });
        // W.random(0.0f, 1.0f);
        W.randn(0.0f, 1.0f / (float) Math.sqrt(in_channels * kernel_size[0] * kernel_size[1]));
        W.requires_grad(true);
        this.params.add(W);

        if (bias) {
            Tensor b = new Tensor(new int[] { out_channels });
            // b.random(0.0f, 1.0f);
            b.randn(0.0f, 1.0f / (float) Math.sqrt(in_channels * kernel_size[0] * kernel_size[1]));
            b.requires_grad(true);
            this.params.add(b);
        }
    }

    public Tensor forward(Tensor input) {
        Tensor w = this.params.get(0);
        Tensor b, out;
        if (this.bias) {
            b = this.params.get(1);
            out = F.conv2d(input, w, b, this.stride, this.padding);
        } else {
            out = F.conv2d(input, w, this.stride, this.padding);
        }
        return out;
    }
}
