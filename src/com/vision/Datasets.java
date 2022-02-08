package com.vision;

import com.data.Tensor;
import com.utils.*;
import com.utils.DataLoader;
import java.nio.file.Paths;

public class Datasets {
        public static DataLoader[] loadDIGITS(String root, int batchSize) {
                DataLoader[] output = new DataLoader[2];
                Tensor x_train = Misc.loadTensor(Paths.get(root,
                                "x_train_digits.bin").toString());
                Tensor y_train = Misc.loadTensor(Paths.get(
                                root,
                                "y_train_digits.bin").toString());
                Tensor x_test = Misc.loadTensor(Paths.get(
                                root,
                                "x_test_digits.bin").toString());
                Tensor y_test = Misc.loadTensor(Paths.get(
                                root,
                                "y_test_digits.bin").toString());

                x_train = x_train.div(new Tensor(16.0f));
                x_test = x_test.div(new Tensor(16.0f));

                Dataset train_ds = new TensorDataset(x_train, y_train);
                DataLoader train_dl = new DataLoader(train_ds, batchSize, true);

                Dataset test_ds = new TensorDataset(x_test, y_test);
                DataLoader test_dl = new DataLoader(test_ds, batchSize, true);

                output[0] = train_dl;
                output[1] = test_dl;
                return output;
        }

        public static DataLoader[] loadMNIST(String root, int batchSize) {
                DataLoader[] output = new DataLoader[2];
                Tensor x_train = Misc.loadTensor(Paths.get(root,
                                "x_train_mnist.bin").toString());
                Tensor y_train = Misc.loadTensor(Paths.get(
                                root,
                                "y_train_mnist.bin").toString());
                Tensor x_test = Misc.loadTensor(Paths.get(
                                root,
                                "x_test_mnist.bin").toString());
                Tensor y_test = Misc.loadTensor(Paths.get(
                                root,
                                "y_test_mnist.bin").toString());

                x_train = x_train.div(new Tensor(255.0f));
                x_test = x_test.div(new Tensor(255.0f));

                Dataset train_ds = new TensorDataset(x_train, y_train);
                DataLoader train_dl = new DataLoader(train_ds, batchSize, true);

                Dataset test_ds = new TensorDataset(x_test, y_test);
                DataLoader test_dl = new DataLoader(test_ds, batchSize, true);

                output[0] = train_dl;
                output[1] = test_dl;
                return output;
        }
}