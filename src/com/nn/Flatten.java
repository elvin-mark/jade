package com.nn;

import com.data.Tensor;

public class Flatten extends NNModule {
    public Flatten() {
        super();
        this.moduleName = "Flatten";
    }

    public Tensor forward(Tensor x) {
        int input_size = x.size;
        int N = x.shape[0];
        int M = input_size / N;
        Tensor output = new Tensor(x);
        output = x.reshape(new int[] { N, M });
        return output;
    }
}
