package com.data;

public class DivBackward extends Node {
    public DivBackward(Tensor t1, Tensor t2, Tensor result) {
        super();
        this.add_child(t1.node);
        this.add_child(t2.node);
        this.tensor = result;
    }

    public void backward(Tensor loss) {
        Tensor t1 = this.children.get(0).tensor;
        Tensor t2 = this.children.get(1).tensor;
        Tensor new_loss_1 = new Tensor(t1.shape);
        Tensor new_loss_2 = new Tensor(t2.shape);
        new_loss_1.zeros();
        new_loss_2.zeros();
        // Get the gradient of the loss with respect to the input t1
        if (t1.size == t2.size) {
            for (int i = 0; i < loss.data.length; i++) {
                new_loss_1.data[i] += loss.data[i] / t2.data[i];
                new_loss_2.data[i] += -t1.data[i] * loss.data[i] / Math.pow(t2.data[i], 2.0f);
            }
        } else {
            int[] new_stride = t1.compare_shapes(t2);
            for (int i = 0; i < loss.data.length; i++) {
                int[] indices = t1.get_indices(i);
                int j = 0;
                for (int k = 0; k < new_stride.length; k++) {
                    j += indices[k] * new_stride[k];

                }
                new_loss_1.data[i] += loss.data[i] / t2.data[j];
                new_loss_2.data[j] += -t1.data[i] * loss.data[i] / Math.pow(t2.data[j], 2.0f);
            }
        }
        t1.node.backward(new_loss_1);
        t2.node.backward(new_loss_2);
    }
}
