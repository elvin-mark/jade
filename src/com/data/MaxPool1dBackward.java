package com.data;

public class MaxPool1dBackward extends Node {
    int kernel;

    public MaxPool1dBackward(Tensor t1, Tensor result, int kernel) {
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
        int L = (t1.shape[2] / this.kernel) * this.kernel;
        int[] indices1, indices2;
        new_loss.zeros();

        // TODO: Check and Change this
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < C; j++) {
                for (int k = 0; k < L; k++) {
                    indices1 = new int[] { i, j, k };
                    indices2 = new int[] { i, j, k / this.kernel };
                    if (t1.at(indices1) == this.tensor.at(indices2)) {
                        new_loss.set(indices1, loss.at(indices2));
                    }

                }
            }
        }

        t1.node.backward(new_loss);
    }
}
