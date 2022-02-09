package com.utils;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.util.Random;
import java.io.File;
import com.data.Tensor;
import com.nn.*;
import com.optim.*;

public class Misc {
    public static final int TYPE_CHAR = 0;
    public static final int TYPE_INT = 1;
    public static final int TYPE_FLOAT = 2;

    public static Tensor loadTensor(String path) {
        try {
            FileInputStream fis = new FileInputStream(new File(path));
            byte[] buffer = new byte[4];
            byte[] data_buffer = new byte[1];
            fis.read(buffer);
            int typeData = ByteBuffer.wrap(buffer).getInt();
            fis.read(buffer);
            int num_shape = ByteBuffer.wrap(buffer).getInt();
            int[] shape = new int[num_shape];
            for (int i = 0; i < num_shape; i++) {
                fis.read(buffer);
                shape[i] = ByteBuffer.wrap(buffer).getInt();
            }
            Tensor out = new Tensor(shape);
            for (int i = 0; i < out.size; i++) {
                if (typeData == TYPE_CHAR) {
                    fis.read(data_buffer);
                    out.data[i] = (float) data_buffer[0];
                } else if (typeData == TYPE_INT) {
                    fis.read(buffer);
                    out.data[i] = ByteBuffer.wrap(buffer).getInt();
                } else if (typeData == TYPE_FLOAT) {
                    fis.read(buffer);
                    out.data[i] = ByteBuffer.wrap(buffer).getFloat();
                }
            }
            return out;
        } catch (Exception e) {
            System.out.println("Error reading file: " + path);
        }
        return null;
    }

    public static void saveTensor(Tensor t, String path) {
        try {
            FileOutputStream fos = new FileOutputStream(new File(path));
            fos.write(ByteBuffer.allocate(4).putInt(TYPE_FLOAT).array());
            fos.write(ByteBuffer.allocate(4).putInt(t.shape.length).array());
            for (int i = 0; i < t.shape.length; i++) {
                fos.write(ByteBuffer.allocate(4).putInt(t.shape[i]).array());
            }
            for (int i = 0; i < t.size; i++) {
                fos.write(ByteBuffer.allocate(4).putFloat(t.data[i]).array());
            }
        } catch (Exception e) {
            System.out.println("Error writing file: " + path);
        }
    }

    public static float[] train_one_epoch(NNModule model, DataLoader train_dl, Optimizer optim, Loss loss_fn) {
        float[] record = new float[2];
        float loss_record = 0.0f;
        float corrects = 0.0f;
        float total = 0.0f;
        model.train();
        for (Tensor[] xy : train_dl) {
            Tensor y_pred = model.forward(xy[0]);
            Tensor loss = loss_fn.criterion(y_pred, xy[1]);
            loss_record += loss.item();
            corrects += y_pred.argmax(1).equal(xy[1]).sum().item();
            total += xy[1].size;
            optim.zero_grad();
            loss.backward();
            optim.step();
        }
        record[0] = loss_record / train_dl.size();
        record[1] = 100 * corrects / total;
        return record;
    }

    public static float[] eval(NNModule model, DataLoader test_dl, Loss loss_fn) {
        float[] record = new float[2];
        float loss_record = 0.0f;
        float corrects = 0.0f;
        float total = 0.0f;
        model.eval();
        for (Tensor[] xy : test_dl) {
            Tensor y_pred = model.forward(xy[0]);
            Tensor loss = loss_fn.criterion(y_pred, xy[1]);
            loss_record += loss.item();
            corrects += y_pred.argmax(1).equal(xy[1]).sum().item();
            total += xy[1].size;
        }
        record[0] = loss_record / test_dl.size();
        record[1] = 100 * corrects / total;
        return record;
    }

    public static void train(NNModule model, DataLoader train_dl, DataLoader test_dl, Optimizer optim, Loss loss_fn,
            int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            float[] record = train_one_epoch(model, train_dl, optim, loss_fn);
            if (test_dl != null) {
                float[] test_record = eval(model, test_dl, loss_fn);
                System.out.println(
                        "Epoch: " + epoch + " Train Loss: " + record[0] + " Train Acc: " + record[1]
                                + " Test Loss: " + test_record[0] + " Test Acc: " + test_record[1]);
            } else {
                System.out.println("Epoch: " + epoch + " Train Loss: " + record[0] + " Train Acc: " + record[1]);
            }
        }
    }

    public static NNModule activation_function(String act_fun) {
        if (act_fun.equals("relu")) {
            return new ReLU();
        } else if (act_fun.equals("sigmoid")) {
            return new Sigmoid();
        } else if (act_fun.equals("tanh")) {
            return new Tanh();
        } else if (act_fun.equals("softmax")) {
            return new Softmax();
        } else {
            System.out.println("Unknown activation function: " + act_fun);
            return null;
        }
    }

    public static NNModule build_mlp(int[] layers, String act_fun, boolean use_bias) {
        Sequential model = new Sequential();
        for (int i = 0; i < layers.length - 2; i++) {
            model.add_module(new Linear(layers[i], layers[i + 1], use_bias));
            model.add_module(activation_function(act_fun));
        }
        model.add_module(new Linear(layers[layers.length - 2], layers[layers.length - 1], use_bias));
        return model;
    }

    public static Tensor[] generate_clusters(int num_clusters, int num_points, int dim) {
        Tensor[] out = new Tensor[2];
        Random rand = new Random();
        out[0] = new Tensor(new int[] { num_points, dim });
        out[1] = new Tensor(new int[] { num_points, 1 });

        Tensor centers = new Tensor(new int[] { num_clusters, dim });
        centers.random(-5.0f, 5.0f);

        for (int i = 0; i < num_points; i++) {
            int cluster_id = rand.nextInt(num_clusters);
            for (int j = 0; j < dim; j++) {
                out[0].data[i * dim + j] = centers.data[cluster_id * dim + j] + (float) rand.nextGaussian() * 0.1f;
            }
            out[1].data[i] = (float) cluster_id;
        }
        return out;
    }
}