package com.utils;

import com.data.Tensor;

public class TensorDataset extends Dataset {
    int[] x_shape, y_shape;
    Tensor x_raw, y_raw;

    public TensorDataset(Tensor x, Tensor y) {
        super();
        if (x.shape[0] != y.shape[0]) {
            System.out.println("Error: x and y should have the same shape.");
            System.exit(1);
        }
        this.length = x.shape[0];
        this.x_shape = new int[x.shape.length - 1];
        this.y_shape = new int[y.shape.length - 1];

        for (int i = 0; i < x_shape.length; i++) {
            x_shape[i] = x.shape[i + 1];
        }
        for (int i = 0; i < y_shape.length; i++) {
            y_shape[i] = y.shape[i + 1];
        }

        this.x_raw = x;
        this.y_raw = y;
    }

    public Tensor[] get(int index) {
        Tensor[] out = new Tensor[2];

        out[0] = new Tensor(x_shape);
        out[1] = new Tensor(y_shape);

        for (int i = 0; i < out[0].size; i++) {
            out[0].data[i] = x_raw.data[index * out[0].size + i];
        }

        for (int i = 0; i < out[1].size; i++) {
            out[1].data[i] = y_raw.data[index * out[1].size + i];
        }

        return out;
    }

    public int[][] get_items_shape() {
        int[][] out = new int[2][];
        out[0] = x_shape;
        out[1] = y_shape;
        return out;
    }
}