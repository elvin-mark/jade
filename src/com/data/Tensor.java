package com.data;

public class Tensor {
    public int[] shape;
    public float[] data;
    public int size;
    public int[] stride;
    public Tensor grad = null;
    public Node node = null;
    public boolean requires_grad_ = false;

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

        if (this.data.length != this.size) {
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
        // int j = 1;

        // for (int k = index.length - 1; k >= 0; k--) {
        // i += index[k] * j;
        // j *= shape[k];
        // }

        for (int k = 0; k < index.length; k++) {
            i += index[k] * this.stride[k];
        }
        return data[i];
    }

    public void set(int[] index, float value) {
        if (index.length != this.shape.length) {
            throw new RuntimeException("Tensor index size mismatch");
        }

        int i = 0;
        // int j = 1;
        // for (int k = index.length - 1; k >= 0; k--) {
        // i += index[k] * j;
        // j *= shape[k];
        // }
        for (int k = 0; k < index.length; k++) {
            i += index[k] * this.stride[k];
        }
        data[i] = value;
    }

    public float item() {
        if (this.shape.length != 1 || this.shape[0] != 1) {
            throw new RuntimeException("Tensor shape mismatch");
        }
        return data[0];
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

    public void reshape(int[] shape) {
        this.view(shape);
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

    public Tensor add(Tensor other) {
        if (this.size % other.size != 0) {
            throw new RuntimeException("Tensor.add: size mismatch");
        }
        Tensor result = new Tensor(this.shape);
        for (int i = 0; i < this.size; i++) {
            result.data[i] = this.data[i] + other.data[i % other.size];
        }
        if (this.requires_grad_ || other.requires_grad_) {
            result.requires_grad(true);
            result.node = new AddBackward(this, other, result);
        }
        return result;
    }

    public Tensor sub(Tensor other) {
        if (this.size % other.size != 0) {
            throw new RuntimeException("Tensor.sub: size mismatch");
        }
        Tensor result = new Tensor(this.shape);
        for (int i = 0; i < this.size; i++) {
            result.data[i] = this.data[i] - other.data[i % other.size];
        }
        if (this.requires_grad_ || other.requires_grad_) {
            result.requires_grad(true);
            result.node = new SubBackward(this, other, result);
        }
        return result;
    }

    public Tensor mul(Tensor other) {
        if (this.size % other.size != 0) {
            throw new RuntimeException("Tensor.mul: size mismatch");
        }
        Tensor result = new Tensor(this.shape);
        for (int i = 0; i < this.size; i++) {
            result.data[i] = this.data[i] * other.data[i % other.size];
        }
        if (this.requires_grad_ || other.requires_grad_) {
            result.requires_grad(true);
            result.node = new MulBackward(this, other, result);
        }
        return result;
    }

    public Tensor div(Tensor other) {
        if (this.size % other.size != 0) {
            throw new RuntimeException("Tensor.div: size mismatch");
        }
        Tensor result = new Tensor(this.shape);
        for (int i = 0; i < this.size; i++) {
            result.data[i] = this.data[i] / other.data[i % other.size];
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
}