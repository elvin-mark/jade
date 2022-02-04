package com.data;

public class LogSoftmaxBackward extends Node {
    public LogSoftmaxBackward(Tensor t1, Tensor result) {
        super();
        this.add_child(t1.node);
        this.tensor = result;
    }

    public void backward(Tensor loss) {
        /*
         * Tensor t: [N,C,...]
         * Tensor tensor: [N,C,...]
         */
        Tensor t = this.children.get(0).tensor;
        Tensor bp_loss = new Tensor(t.shape);
        // Get the gradient of the loss with respect to the input
        int num_batch = t.shape[0];
        int num_classes = t.shape[1];

        // Loop over batches
        for (int i = 0; i < this.tensor.shape[0]; i++) {
            for (int j = 0; j < this.tensor.size / (num_batch * num_classes); j++) {
                float sum = 0;
                for (int k = 0; k < num_classes; k++) {
                    bp_loss.data[i * t.stride[0] + k * t.stride[1] + j] = loss.data[i * t.stride[0]
                            + k * t.stride[1] + j];
                    sum += bp_loss.data[i * t.stride[0] + k * t.stride[1] + j];
                }
                for (int k = 0; k < num_classes; k++) {
                    bp_loss.data[i * t.stride[0] + k * t.stride[1]
                            + j] -= ((float) Math.exp(this.tensor.data[i * t.stride[0] + k * t.stride[1] + j])) * sum;
                }
            }
        }
        t.node.backward(bp_loss);
    }
}
