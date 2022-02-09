package com.data;

import java.util.ArrayList;

public class Tensor {
    public int[] shape;
    public float[] data;
    public int size;
    public int[] stride;
    public Tensor grad = null;
    public Node node = null;
    public boolean requires_grad_ = false;
    int offset = 0;

    // Constructors
    public Tensor(int[] shape) {
        this.shape = shape;
        this.stride = new int[shape.length];
        this.size = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            this.stride[i] = this.size;
            this.size *= shape[i];
        }
        this.data = new float[this.size];
        this.node = new Node(this);
    }

    public Tensor(int[] shape, float[] data) {
        this.shape = shape;
        this.stride = new int[shape.length];
        this.size = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            this.stride[i] = this.size;
            this.size *= shape[i];
        }

        if (data.length != this.size) {
            throw new RuntimeException("Tensor data size mismatch");
        }
        this.data = data;
        this.node = new Node(this);
    }

    public Tensor(Tensor tensor) {
        this.shape = tensor.shape;
        this.size = tensor.size;
        this.data = tensor.data;
        this.stride = tensor.stride;
        this.offset = tensor.offset;
        this.node = new Node(this);
    }

    public Tensor(float[] data) {
        this.shape = new int[] { data.length };
        this.size = data.length;
        this.data = data;
        this.stride = new int[] { 1 };
        this.node = new Node(this);
    }

    public Tensor(float value) {
        this.shape = new int[] { 1 };
        this.size = 1;
        this.data = new float[] { value };
        this.stride = new int[] { 1 };
        this.node = new Node(this);
    }

    public Tensor clone() {
        return new Tensor(this);
    }

    // Useful intializers
    public void ones() {
        for (int i = 0; i < this.size; i++) {
            this.data[i] = 1.0f;
        }
    }

    public void zeros() {
        for (int i = 0; i < this.size; i++) {
            this.data[i] = 0.0f;
        }
    }

    public void random(float min, float max) {
        for (int i = 0; i < this.size; i++) {
            this.data[i] = (float) (Math.random() * (max - min) + min);
        }
    }

    public void randn(float mean, float std) {
        for (int i = 0; i < this.size; i++) {
            this.data[i] = (float) (Math.random() * std + mean);
        }
    }

    // Printing helper
    public void print() {
        // TODO: print tensor

        // Modidy this snippet to print tensor
        for (int i = 0; i < this.data.length; i++) {
            System.out.print(this.data[i] + " ");
        }
    }

    public String toString() {
        // TODO: print tensor
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < this.data.length; i++) {
            sb.append(this.data[i] + " ");
        }
        return sb.toString();
    }

    // Data accessors and view methods
    public float at(int[] index) {
        if (index.length != this.shape.length) {
            throw new RuntimeException("Tensor index size mismatch");
        }

        int i = 0;

        for (int k = 0; k < index.length; k++) {
            i += index[k] * this.stride[k];
        }
        return data[i + this.offset];
    }

    public void set(int[] index, float value) {
        if (index.length != this.shape.length) {
            throw new RuntimeException("Tensor index size mismatch");
        }

        int i = 0;

        for (int k = 0; k < index.length; k++) {
            i += index[k] * this.stride[k];
        }
        data[i + this.offset] = value;
    }

    public float item() {
        if (this.shape.length != 1 || this.shape[0] != 1) {
            throw new RuntimeException("Tensor shape mismatch");
        }
        return data[0];
    }

    public int[] get_indices(int index) {
        int[] indices = new int[this.shape.length];
        int i = index;
        for (int k = 0; k < indices.length; k++) {
            indices[k] = i / this.stride[k];
            i = i % this.stride[k];
        }
        return indices;
    }

    public int[] compare_shapes(Tensor other) {
        if (this.size % other.size != 0)
            throw new RuntimeException("Tensor shape mismatch");
        int[] new_stride = new int[this.stride.length];
        for (int i = 0; i < this.stride.length; i++) {
            if (i < other.stride.length) {
                if (this.shape[i] == other.shape[i]) {
                    new_stride[i] = other.stride[i];
                } else if (other.shape[i] == 1) {
                    new_stride[i] = 0;
                } else {
                    throw new RuntimeException("Tensor shape mismatch");
                }
            } else {
                new_stride[i] = 0;
            }
        }
        return new_stride;
    }

    public void view(int[] shape) {
        int new_size = 1;
        int[] new_stride = new int[shape.length];

        for (int i = shape.length - 1; i >= 0; i--) {
            new_stride[i] = new_size;
            new_size *= shape[i];
        }

        if (this.size != new_size)
            throw new RuntimeException("Tensor size mismatch");

        this.shape = shape;
        this.size = new_size;
        this.stride = new_stride;
    }

    public Tensor reshape(int[] new_shape) {
        Tensor result = new Tensor(this);
        result.view(new_shape);
        if (this.requires_grad_) {
            result.requires_grad(true);
            result.node = new ReshapeBackward(this, result);
        }
        return result;
    }

    public Tensor transpose() {
        // Add TransposeBackward?
        Tensor result = new Tensor(this);
        int[] new_stride = new int[this.shape.length];
        int[] new_shape = new int[this.shape.length];
        for (int i = 0; i < this.shape.length; i++) {
            new_stride[i] = this.stride[this.shape.length - i - 1];
            new_shape[i] = this.shape[this.shape.length - i - 1];
        }
        result.stride = new_stride;
        result.shape = new_shape;
        return result;
    }

    public int[] shape() {
        return this.shape;
    }

    public int[] stride() {
        return this.stride;
    }

    public Tensor argmax(int axis) {
        // TODO: fix this
        // No gradients for now
        if (axis >= this.shape.length) {
            throw new RuntimeException("Tensor index size mismatch");
        }
        int[] new_shape = new int[this.shape.length - 1];
        for (int i = 0; i < new_shape.length; i++) {
            if (i < axis)
                new_shape[i] = this.shape[i];
            else
                new_shape[i] = this.shape[i + 1];
        }
        Tensor out = new Tensor(new_shape);
        int skip_stride = this.stride[axis];
        int max_index = 0;
        for (int i = 0; i < out.size; i++) {
            max_index = 0;
            for (int j = 1; j < this.shape[axis]; j++) {
                if (this.data[i + j * skip_stride] > this.data[i + max_index * skip_stride])
                    max_index = j;
            }
            out.data[i] = (float) max_index;
        }
        return out;
    }

    public Tensor sub_tensor(int[] index) {
        if (index.length != this.shape.length) {
            throw new RuntimeException("Tensor index size mismatch");
        }
        ArrayList<Integer> new_shape = new ArrayList<Integer>();
        ArrayList<Integer> new_stride = new ArrayList<Integer>();
        int new_size = 1;
        Tensor result = new Tensor(this);
        int i = 0;
        for (int k = 0; k < index.length; k++) {
            if (index[k] < 0) {
                new_size *= this.shape[k];
                new_shape.add(this.shape[k]);
                new_stride.add(this.stride[k]);
            } else
                i += index[k] * this.stride[k];
        }
        result.offset = i;
        result.size = new_size;
        result.shape = new int[new_shape.size()];
        for (Integer elem : new_shape)
            result.shape[i] = new_shape.get(elem);
        result.stride = new int[new_stride.size()];
        for (Integer elem : new_stride)
            result.stride[i] = new_stride.get(elem);
        return result;
    }

    // Gradient related functions
    public void set_grad(Tensor grad) {
        if (grad.size != this.size) {
            throw new RuntimeException("Tensor size mismatch");
        }
        this.grad = grad;
    }

    public void set_data(float[] data) {
        if (data.length != this.data.length) {
            throw new RuntimeException("Tensor data size mismatch");
        }
        this.data = data;
    }

    public void zero_grad() {
        if (this.grad != null) {
            this.grad.zeros();
        }
    }

    public void requires_grad(boolean requires_grad) {
        this.requires_grad_ = requires_grad;
        if (this.requires_grad_) {
            this.grad = new Tensor(this.shape);
            this.grad.zeros();
            this.node = new Node(this);
        } else {
            this.grad = null;
            this.node = null;
        }
    }

    public void backward(Tensor loss) {
        if (this.requires_grad_) {
            this.node.backward(loss);
        }
    }

    public void backward() {
        if (this.requires_grad_) {
            this.node.backward(new Tensor(1.0f));
        }

    }

    // Tensor operations
    public Tensor mm(Tensor other) {
        if (this.shape.length != 2 || other.shape.length != 2) {
            throw new RuntimeException("Tensor.mm: only support matrix multiplication");
        }
        if (this.shape[1] != other.shape[0]) {
            throw new RuntimeException(
                    "Tensor.mm: matrix multiplication requires the number of columns of the first tensor to be equal to the number of rows of the second tensor");
        }
        int[] shape = new int[] { this.shape[0], other.shape[1] };
        Tensor result = new Tensor(shape);
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                float sum = 0;
                for (int k = 0; k < this.shape[1]; k++) {
                    sum += this.at(new int[] { i, k }) * other.at(new int[] { k, j });
                }
                result.set(new int[] { i, j }, sum);
            }
        }
        if (this.requires_grad_ || other.requires_grad_) {
            result.requires_grad(true);
            result.node = new MmBackward(this, other, result);
        }
        return result;
    }

    public Tensor conv2d(Tensor other, int[] stride, int[] padding) {
        // shape of tensor: [N, Cin, H, W]
        // shape of kernel: [Cout, Cin, K1, K2]
        if (this.shape.length != 4 || other.shape.length != 4) {
            throw new RuntimeException("Tensor.conv2d: only support convolution on 4D tensor");
        }
        if (this.shape[1] != other.shape[1]) {
            throw new RuntimeException(
                    "Tensor.conv2d: convolution requires the number of channels of the first tensor to be equal to the number of channels of the second tensor");
        }
        int N = this.shape[0];
        int Cin = this.shape[1];
        int H = this.shape[2];
        int W = this.shape[3];
        int K1 = other.shape[2];
        int K2 = other.shape[3];
        int Cout = other.shape[0];
        int new_H = (H - K1 + 2 * padding[0]) / stride[0] + 1;
        int new_W = (W - K2 + 2 * padding[1]) / stride[1] + 1;
        int[] new_shape = new int[] { N, Cout, new_H, new_W };
        int raw_h, raw_w;
        Tensor result = new Tensor(new_shape);
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < Cout; c++) {
                for (int h = 0; h < new_H; h++) {
                    for (int w = 0; w < new_W; w++) {
                        float sum = 0;
                        for (int c1 = 0; c1 < Cin; c1++) {
                            for (int k1 = 0; k1 < K1; k1++) {
                                for (int k2 = 0; k2 < K2; k2++) {
                                    raw_h = h * stride[0] + k1 - padding[0];
                                    raw_w = w * stride[1] + k2 - padding[1];
                                    if (raw_h >= 0 && raw_h < H && raw_w >= 0 && raw_w < W) {
                                        sum += this.at(new int[] { n, c1, raw_h, raw_w })
                                                * other.at(new int[] { c, c1, k1, k2 });
                                    }
                                }
                            }
                        }
                        result.set(new int[] { n, c, h, w }, sum);
                    }
                }
            }
        }

        if (this.requires_grad_ || other.requires_grad_) {
            result.requires_grad(true);
            result.node = new Conv2dBackward(this, other, result, stride, padding);
        }

        return result;
    }

    public Tensor conv1d(Tensor other, int stride, int padding) {
        // shape of tensor: [N, Cin, L]
        // shape of kernel: [Cout, Cin, K]

        if (this.shape.length != 3 || other.shape.length != 3) {
            throw new RuntimeException("Tensor.conv1d: only support convolution on 3D tensor");
        }
        int N = this.shape[0];
        int Cin = this.shape[1];
        int L = this.shape[2];
        int Cout = other.shape[0];
        int K = other.shape[2];

        int raw_k;

        if (Cin != other.shape[1]) {
            throw new RuntimeException(
                    "Tensor.conv1d: convolution requires the number of channels of the first tensor to be equal to the number of channels of the second tensor");
        }
        int new_L = (L - K + 2 * padding) / stride + 1;
        int[] new_shape = new int[] { N, Cout, new_L };

        Tensor result = new Tensor(new_shape);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < Cout; j++) {
                for (int k = 0; k < new_L; k++) {
                    float sum = 0;
                    for (int p = 0; p < Cin; p++) {
                        for (int q = 0; q < K; q++) {
                            raw_k = k * stride + q - padding;
                            if (raw_k >= 0 && raw_k < L) {
                                sum += this.at(new int[] { i, p, raw_k }) * other.at(new int[] { j, p, q });
                            }
                        }
                    }
                    result.set(new int[] { i, j, k }, sum);
                }
            }
        }

        if (this.requires_grad_ || other.requires_grad_) {
            result.requires_grad(true);
            result.node = new Conv1dBackward(this, other, result, stride, padding);
        }

        return result;
    }

    public Tensor maxpool2d(int[] kernel) {
        int[] new_shape = new int[this.shape.length];
        new_shape[2] = this.shape[2] / kernel[0];
        new_shape[3] = this.shape[3] / kernel[1];
        int N = this.shape[0];
        int Cin = this.shape[1];
        new_shape[0] = N;
        new_shape[1] = Cin;

        int H = new_shape[2] * kernel[0];
        int W = new_shape[3] * kernel[1];

        Tensor result = new Tensor(new_shape);

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < Cin; j++) {
                for (int k = 0; k < H; k += kernel[0]) {
                    for (int l = 0; l < W; l += kernel[1]) {
                        float max = this.at(new int[] { i, j, k, l });
                        for (int m = 0; m < kernel[0]; m++) {
                            for (int n = 0; n < kernel[1]; n++) {
                                max = Math.max(max, this.at(new int[] { i, j, k + m, l + n }));
                            }
                        }
                        result.set(new int[] { i, j, k / kernel[0], l / kernel[1] }, max);
                    }
                }
            }
        }

        if (this.requires_grad_) {
            result.requires_grad(true);
            result.node = new MaxPool2dBackward(this, result, kernel);
        }

        return result;
    }

    public Tensor dropout2d(float p) {
        Tensor result = new Tensor(this.shape);
        for (int i = 0; i < this.size; i++) {
            if (Math.random() < p) {
                result.data[i] = 0;
            } else {
                result.data[i] = this.data[i] / p;
            }
        }

        if (this.requires_grad_) {
            result.requires_grad(true);
            result.node = new Dropout2dBackward(this, result, p);
        }
        return result;
    }

    public Tensor add(Tensor other) {
        int[] new_stride = this.compare_shapes(other);

        Tensor result = new Tensor(this.shape);

        if (this.size == other.size) {
            for (int i = 0; i < this.size; i++) {
                result.data[i] = this.data[i] + other.data[i];
            }
        }

        else {
            for (int i = 0; i < this.size; i++) {
                int[] indices = this.get_indices(i);
                int j = 0;
                for (int k = 0; k < new_stride.length; k++) {
                    j += indices[k] * new_stride[k];
                }
                result.data[i] = this.data[i] + other.data[j];
            }
        }

        if (this.requires_grad_ || other.requires_grad_) {
            result.requires_grad(true);
            result.node = new AddBackward(this, other, result);
        }
        return result;
    }

    public Tensor sub(Tensor other) {
        int[] new_stride = this.compare_shapes(other);

        Tensor result = new Tensor(this.shape);

        if (this.size == other.size) {
            for (int i = 0; i < this.size; i++) {
                result.data[i] = this.data[i] - other.data[i];
            }
        }

        else {
            for (int i = 0; i < this.size; i++) {
                int[] indices = this.get_indices(i);
                int j = 0;
                for (int k = 0; k < new_stride.length; k++) {
                    j += indices[k] * new_stride[k];
                }
                result.data[i] = this.data[i] - other.data[j];
            }
        }
        if (this.requires_grad_ || other.requires_grad_) {
            result.requires_grad(true);
            result.node = new SubBackward(this, other, result);
        }
        return result;
    }

    public Tensor mul(Tensor other) {
        int[] new_stride = this.compare_shapes(other);

        Tensor result = new Tensor(this.shape);

        if (this.size == other.size) {
            for (int i = 0; i < this.size; i++) {
                result.data[i] = this.data[i] * other.data[i];
            }
        }

        else {
            for (int i = 0; i < this.size; i++) {
                int[] indices = this.get_indices(i);
                int j = 0;
                for (int k = 0; k < new_stride.length; k++) {
                    j += indices[k] * new_stride[k];
                }
                result.data[i] = this.data[i] * other.data[j];
            }
        }
        if (this.requires_grad_ || other.requires_grad_) {
            result.requires_grad(true);
            result.node = new MulBackward(this, other, result);
        }
        return result;
    }

    public Tensor div(Tensor other) {
        int[] new_stride = this.compare_shapes(other);

        Tensor result = new Tensor(this.shape);

        if (this.size == other.size) {
            for (int i = 0; i < this.size; i++) {
                result.data[i] = this.data[i] / other.data[i];
            }
        }

        else {
            for (int i = 0; i < this.size; i++) {
                int[] indices = this.get_indices(i);
                int j = 0;
                for (int k = 0; k < new_stride.length; k++) {
                    j += indices[k] * new_stride[k];
                }
                result.data[i] = this.data[i] / other.data[j];
            }
        }
        if (this.requires_grad_ || other.requires_grad_) {
            result.requires_grad(true);
            result.node = new DivBackward(this, other, result);
        }
        return result;
    }

    public Tensor pow(float exponent) {
        Tensor result = new Tensor(this.shape);
        for (int i = 0; i < this.size; i++) {
            result.data[i] = (float) Math.pow(this.data[i], exponent);
        }
        if (this.requires_grad_) {
            result.requires_grad(true);
            result.node = new PowBackward(this, exponent, result);
        }
        return result;
    }

    public Tensor log() {
        Tensor result = new Tensor(this.shape);
        for (int i = 0; i < this.size; i++) {
            result.data[i] = (float) Math.log(this.data[i]);
        }
        if (this.requires_grad_) {
            result.requires_grad(true);
            result.node = new LogBackward(this, result);
        }
        return result;
    }

    public Tensor exp() {
        Tensor result = new Tensor(this.shape);
        for (int i = 0; i < this.size; i++) {
            result.data[i] = (float) Math.exp(this.data[i]);
        }
        if (this.requires_grad_) {
            result.requires_grad(true);
            result.node = new ExpBackward(this, result);
        }
        return result;
    }

    public Tensor mean() {
        Tensor result = new Tensor(new int[] { 1 });
        for (int i = 0; i < this.size; i++) {
            result.data[0] += this.data[i];
        }
        result.data[0] /= this.size;

        if (this.requires_grad_) {
            result.requires_grad(true);
            result.node = new MeanBackward(this, result);
        }
        return result;
    }

    public Tensor norm() {
        Tensor result = new Tensor(new int[] { 1 });
        for (int i = 0; i < this.size; i++) {
            result.data[0] += this.data[i] * this.data[i];
        }
        result.data[0] = (float) Math.sqrt(result.data[0]);

        if (this.requires_grad_) {
            result.requires_grad(true);
            result.node = new NormBackward(this, result);
        }
        return result;
    }

    public Tensor einsum(String equation, Tensor other) {
        // FIXME
        // String[] terms = equation.split("\\,");
        // if (terms.length != 2) {
        // throw new RuntimeException("Tensor.einsum: only support two terms");
        // }
        // String[] term1 = terms[0].split("\\*");
        // String[] term2 = terms[1].split("\\*");
        // if (term1.length != term2.length) {
        // throw new RuntimeException("Tensor.einsum: terms must have the same number of
        // dimensions");
        // }
        // int[] shape = new int[term1.length];
        // for (int i = 0; i < term1.length; i++) {
        // if (term1[i].equals(term2[i])) {
        // shape[i] = this.shape[i];
        // } else {
        // shape[i] = this.shape[get_index(term1[i])] * this.shape[get_index(term2[i])];
        // }
        // }
        // Tensor result = new Tensor(shape);
        // for (int i = 0; i < result.size; i++) {
        // int[] index = get_index_array(i, result.shape);
        // int[] index1 = get_index_array(i, this.shape);
        // int[] index2 = get_index_array(i, other.shape);
        // result.data[i] = this.at(index1) * other.at(index2);
        // }
        // if (this.requires_grad_ || other.requires_grad_) {
        // result.requires_grad(true);
        // result.node = new EinsumBackward(this, other, result, equation);
        // }
        // return result;
        return null;
    }

    public Tensor sigmoid() {
        Tensor result = new Tensor(this.shape);
        for (int i = 0; i < this.size; i++) {
            result.data[i] = (float) (1 / (1 + Math.exp(-this.data[i])));
        }
        if (this.requires_grad_) {
            result.requires_grad(true);
            result.node = new SigmoidBackward(this, result);
        }
        return result;
    }

    public Tensor tanh() {
        Tensor result = new Tensor(this.shape);
        for (int i = 0; i < this.size; i++) {
            result.data[i] = (float) Math.tanh(this.data[i]);
        }
        if (this.requires_grad_) {
            result.requires_grad(true);
            result.node = new TanhBackward(this, result);
        }
        return result;
    }

    public Tensor relu() {
        Tensor result = new Tensor(this.shape);
        for (int i = 0; i < this.size; i++) {
            result.data[i] = Math.max(0, this.data[i]);
        }
        if (this.requires_grad_) {
            result.requires_grad(true);
            result.node = new ReluBackward(this, result);
        }
        return result;
    }

    public Tensor leaky_relu(float alpha) {
        Tensor result = new Tensor(this.shape);
        for (int i = 0; i < this.size; i++) {
            result.data[i] = Math.max(alpha * this.data[i], this.data[i]);
        }
        if (this.requires_grad_) {
            result.requires_grad(true);
            result.node = new LeakyReluBackward(this, result, alpha);
        }
        return result;
    }

    public Tensor softmax() {
        // input: [N, C , ...]
        // output: [N, C , ...]

        Tensor result = new Tensor(this.shape);
        int num_batch = this.shape[0];
        int num_classes = this.shape[1];

        // Loop over batches
        for (int i = 0; i < this.shape[0]; i++) {
            for (int j = 0; j < this.size / (num_batch * num_classes); j++) {
                float sum = 0;
                for (int k = 0; k < num_classes; k++) {
                    result.data[i * this.stride[0] + k * this.stride[1] + j] = (float) Math
                            .exp(this.data[i * this.stride[0] + k * this.stride[1] + j]);
                    sum += result.data[i * this.stride[0] + k * this.stride[1] + j];
                }
                for (int k = 0; k < num_classes; k++) {
                    result.data[i * this.stride[0] + k * this.stride[1] + j] /= sum;
                }
            }

        }
        if (this.requires_grad_) {
            result.requires_grad(true);
            result.node = new SoftmaxBackward(this, result);
        }
        return result;
    }

    public Tensor logsoftmax() {
        Tensor result = new Tensor(this.shape);
        int num_batch = this.shape[0];
        int num_classes = this.shape[1];

        // Loop over batches
        for (int i = 0; i < this.shape[0]; i++) {
            for (int j = 0; j < this.size / (num_batch * num_classes); j++) {
                float sum = 0;
                for (int k = 0; k < num_classes; k++) {
                    result.data[i * this.stride[0] + k * this.stride[1] + j] = (float) Math
                            .exp(this.data[i * this.stride[0] + k * this.stride[1] + j]);
                    sum += result.data[i * this.stride[0] + k * this.stride[1] + j];
                }
                for (int k = 0; k < num_classes; k++) {
                    result.data[i * this.stride[0] + k * this.stride[1] + j] = (float) Math
                            .log(result.data[i * this.stride[0] + k * this.stride[1] + j] / sum);
                }
            }
        }
        if (this.requires_grad_) {
            result.requires_grad(true);
            result.node = new LogSoftmaxBackward(this, result);
        }
        return result;
    }

    public Tensor nll(Tensor target) {
        int[] targetShape = target.shape();

        if (shape.length < 2 || shape.length != targetShape.length || shape[0] != targetShape[0]) {
            throw new IllegalArgumentException("input and target must have the same shape");
        }

        int batchSize = shape[0];
        int[] targetStride = target.stride();

        float kl_divergence = 0;
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < target.size / batchSize; j++)
                kl_divergence += -this.data[i * stride[0] + ((int) target.data[i * targetStride[0] + j]) * stride[1]
                        + j];
        }

        Tensor output = new Tensor(kl_divergence / target.size);
        if (this.requires_grad_) {
            output.requires_grad(true);
            output.node = new NLLLossBackward(this, target, output);
        }
        return output;
    }

}