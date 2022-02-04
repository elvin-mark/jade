package com.data;

public class NLLLossBackward extends Node {
    public NLLLossBackward(Tensor t1, Tensor t2, Tensor result) {
        super();
        this.add_child(t1.node);
        this.add_child(t2.node);
        this.tensor = result;
    }

    public void backward(Tensor loss) {
        // FIX ME
        /*
         * t1: [N, C, ... ]
         * t2: [N, 1, ... ]
         */
        Tensor t1 = this.children.get(0).tensor;
        Tensor t2 = this.children.get(1).tensor;

        Tensor new_loss = new Tensor(t1.shape);
        new_loss.zeros();

        int batchSize = t1.shape[0];
        int[] targetStride = t2.stride();

        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < t2.size / batchSize; j++)
                new_loss.data[i * new_loss.stride[0] + ((int) t2.data[i * targetStride[0] + j]) * new_loss.stride[1]
                        + j] = -1.0f / t2.size;
        }
        t1.node.backward(new_loss);
        // t2.node.backward(new_loss); we do not backpropagate here
    }
}