package com.data;

public class Conv1dBackward extends Node {
    int stride;
    int padding;

    public Conv1dBackward(Tensor t1, Tensor t2, Tensor result, int stride, int padding) {
        super();
        this.add_child(t1.node);
        this.add_child(t2.node);
        this.stride = stride;
        this.padding = padding;
        this.tensor = result;
    }

    public void backward(Tensor loss) {
        Tensor t1 = this.children.get(0).tensor;
        Tensor t2 = this.children.get(1).tensor;

        Tensor new_loss_1 = new Tensor(t1.shape);
        Tensor new_loss_2 = new Tensor(t2.shape);

        int N = t1.shape[0];
        int Cin = t1.shape[1];
        int L = t1.shape[2];
        int Cout = t2.shape[0];
        int K = t2.shape[2];
        int new_L = this.tensor.shape[2];

        int raw_k;
        // Get the gradient of the loss with respect to the input
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < Cin; j++) {
                for (int k = 0; k < L; k++) {
                    float sum = 0;
                    for (int p = 0; p < Cout; p++) {
                        for (int q = 0; q < new_L; q++) {
                            raw_k = k + this.padding - q * this.stride;
                            if (raw_k >= 0 && raw_k < K) {
                                sum += t2.at(new int[] { p, j, raw_k }) * loss.at(new int[] { i, p, q });
                            }
                        }
                    }
                    new_loss_1.set(new int[] { i, j, k }, sum);
                }
            }
        }
        // Get the gradient of the loss with respect to the weight
        for (int i = 0; i < Cout; i++) {
            for (int j = 0; j < Cin; j++) {
                for (int k = 0; k < K; k++) {
                    float sum = 0;
                    for (int p = 0; p < N; p++) {
                        for (int q = 0; q < new_L; q++) {
                            raw_k = q * this.stride + k - this.padding;
                            if (raw_k >= 0 && raw_k < L) {
                                sum += t1.at(new int[] { p, j, raw_k }) * loss.at(new int[] { p, i, q });
                            }
                        }
                    }
                    new_loss_2.set(new int[] { i, j, k }, sum);
                }
            }
        }

        t1.node.backward(new_loss_1);
        t2.node.backward(new_loss_2);
    }
}