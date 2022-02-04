package com.utils;

import com.data.Tensor;

public class TensorDataset extends Dataset {
    Tensor x_raw, y_raw;

    public TensorDataset(Tensor x, Tensor y) {
        int[] x_shape = x.shape;
        int[] y_shape = y.shape;
        if (x_shape.length != y_shape.length) {
            System.out.println("Error: x and y should have the same shape.");
            System.exit(1);
        }
        this.x_raw = x;
        this.y_raw = y;
    }

    public Tensor[] get(int index) {
        return null;
    }
}
