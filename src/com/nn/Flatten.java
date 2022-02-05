package com.nn;

import com.data.Tensor;

public class Flatten extends NNModule {
    public Flatten() {
        super();
    }

    public Tensor forward(Tensor x) {
        int input_size = x.size;
        int N = x.shape[0];
        int M = input_size / N;
        Tensor output = new Tensor(x);
        output.reshape(new int[] { N, M });
        output.requires_grad_ = x.requires_grad_;
        return output;
    }
}
