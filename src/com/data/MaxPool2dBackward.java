package com.data;

public class MaxPool2dBackward extends Node {
    int[] kernel;

    public MaxPool2dBackward(Tensor t1, Tensor result, int[] kernel) {
        super();
        this.add_child(t1.node);
        this.tensor = result;
        this.kernel = kernel;
    }

    public void backward(Tensor loss) {
        Tensor t1 = this.children.get(0).tensor;
        Tensor new_loss = new Tensor(t1.shape);

        int N = t1.shape[0];
        int C = t1.shape[1];
        int H = (t1.shape[2] / this.kernel[0]) * this.kernel[0];
        int W = (t1.shape[3] / this.kernel[1]) * this.kernel[1];
        int[] indices1, indices2;
        new_loss.zeros();
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < C; j++) {
                for (int k = 0; k < H; k++) {
                    for (int l = 0; l < W; l++) {
                        indices1 = new int[] { i, j, k, l };
                        indices2 = new int[] { i, j, k / this.kernel[0], l / this.kernel[1] };
                        if (t1.at(indices1) == this.tensor.at(indices2)) {
                            new_loss.set(indices1, loss.at(indices2));
                        }
                    }
                }
            }
        }

        t1.node.backward(new_loss);
    }
}
