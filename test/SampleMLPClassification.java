import com.nn.*;
import com.optim.*;
import com.data.*;
import com.utils.*;
import java.util.*;

import java.nio.file.Paths;

public class SampleMLPClassification {
        public static void main(String[] args) {
                Sequential model = new Sequential();
                model.add_module(new Flatten());
                model.add_module(new Linear(64, 32, true));
                model.add_module(new Tanh());
                model.add_module(new Linear(32, 16, true));
                model.add_module(new Tanh());
                model.add_module(new Linear(16, 10, true));

                String path_to_digits = "./data/DIGITS/";

                Tensor x_train = Misc.loadTensor(Paths.get(path_to_digits,
                                "x_train_digits.bin").toString());
                Tensor y_train = Misc.loadTensor(Paths.get(path_to_digits,
                                "y_train_digits.bin").toString());
                Tensor x_test = Misc.loadTensor(Paths.get(path_to_digits,
                                "x_test_digits.bin").toString());
                Tensor y_test = Misc.loadTensor(Paths.get(path_to_digits,
                                "y_test_digits.bin").toString());

                x_train = x_train.div(new Tensor(16.0f));
                x_test = x_test.div(new Tensor(16.0f));

                Dataset train_ds = new TensorDataset(x_train, y_train);
                DataLoader train_dl = new DataLoader(train_ds, 32, true);

                Dataset test_ds = new TensorDataset(x_test, y_test);
                DataLoader test_dl = new DataLoader(test_ds, 32, true);

                HashMap<String, Float> hyperparams = new HashMap<String, Float>();
                hyperparams.put("lr", 0.01f);
                hyperparams.put("momentum", 0.9f);
                Optimizer optim = new Adam(model.parameters(), hyperparams);
                Loss loss_fn = new CrossEntropyLoss();

                Misc.train(model, train_dl, test_dl, optim, loss_fn, 10);
        }
}
