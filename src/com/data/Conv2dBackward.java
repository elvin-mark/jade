package com.data;

public class Conv2dBackward extends Node {
    int[] stride;
    int[] padding;

    public Conv2dBackward(Tensor t1, Tensor t2, Tensor result, int[] stride, int[] padding) {
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
        int H = t1.shape[2];
        int W = t1.shape[3];
        int K1 = t2.shape[2];
        int K2 = t2.shape[3];
        int Cout = t2.shape[0];
        int new_H = this.tensor.shape[2];
        int new_W = this.tensor.shape[3];
        int raw_h, raw_w;

        // Get the gradient of the loss with respect to the input
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < Cin; c++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        float sum = 0;
                        for (int c1 = 0; c1 < Cout; c1++) {
                            for (int h1 = 0; h1 < new_H; h1++) {
                                for (int w1 = 0; w1 < new_W; w1++) {
                                    raw_h = h - h1 * this.stride[0] + this.padding[0];
                                    raw_w = w - w1 * this.stride[1] + this.padding[1];
                                    if (raw_h >= 0 && raw_h < K1 && raw_w >= 0 && raw_w < K2) {
                                        sum += loss.at(new int[] { n, c1, h1, w1 })
                                                * t2.at(new int[] { c1, c, raw_h, raw_w });
                                    }
                                }
                            }
                        }
                        new_loss_1.set(new int[] { n, c, h, w }, sum);
                    }
                }
            }
        }

        // Get the gradient of the loss with respect to the weight
        for (int n = 0; n < Cout; n++) {
            for (int c = 0; c < Cin; c++) {
                for (int h = 0; h < K1; h++) {
                    for (int w = 0; w < K2; w++) {
                        float sum = 0;
                        for (int c1 = 0; c1 < N; c1++) {
                            for (int h1 = 0; h1 < new_H; h1++) {
                                for (int w1 = 0; w1 < new_W; w1++) {
                                    raw_h = h1 * this.stride[0] + h - this.padding[0];
                                    raw_w = w1 * this.stride[1] + w - this.padding[1];
                                    if (raw_h >= 0 && raw_h < H && raw_w >= 0 && raw_w < W) {
                                        sum += loss.at(new int[] { c1, n, h1, w1 })
                                                * t1.at(new int[] { c1, c, raw_h, raw_w });
                                    }
                                }
                            }
                        }
                        new_loss_2.set(new int[] { n, c, h, w }, sum);
                    }
                }
            }
        }

        t1.node.backward(new_loss_1);
        t2.node.backward(new_loss_2);
    }
}