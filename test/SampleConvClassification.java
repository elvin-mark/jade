import com.data.Tensor;
import com.utils.*;
import com.optim.*;
import com.nn.*;

import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

public class SampleConvClassification {
    public static void main(String args[]) {
        Sequential model = new Sequential();

        model.add_module((NNModule) new Conv2d(1, 8, 3, true));
        model.add_module((NNModule) new Sigmoid());
        model.add_module((NNModule) new MaxPool2d(2));
        model.add_module((NNModule) new Conv2d(8, 16, 2, true));
        model.add_module((NNModule) new Sigmoid());
        model.add_module((NNModule) new MaxPool2d(2));
        model.add_module((NNModule) new Flatten());
        model.add_module((NNModule) new Linear(16, 10, true));

        String path_to_digits = System.getenv().get("PATH_TO_DIGITS");

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
        DataLoader train_dl = new DataLoader(train_ds, 64, true);

        Dataset test_ds = new TensorDataset(x_test, y_test);
        DataLoader test_dl = new DataLoader(test_ds, 64, true);

        HashMap<String, Float> hyperparams = new HashMap<String, Float>();
        hyperparams.put("lr", 1.0f);
        Optimizer optim = new SGD(model.parameters(), hyperparams);
        Loss loss_fn = new CrossEntropyLoss();

        Misc.train(model, train_dl, test_dl, optim, loss_fn, 20);
    }

}
