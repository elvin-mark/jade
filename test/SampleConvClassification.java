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
                model.add_module((NNModule) new Conv2d(8, 16, 3, true));
                model.add_module((NNModule) new Sigmoid());
                model.add_module((NNModule) new Flatten());
                model.add_module((NNModule) new Linear(16, 10, true));

                String path_to_digits = System.getenv().get("PATH_TO_DIGITS");

                DataLoader[] dl = com.vision.Datasets.loadDigits(path_to_digits, 64);

                HashMap<String, Float> hyperparams = new HashMap<String, Float>();
                hyperparams.put("lr", 1.0f);
                Optimizer optim = new SGD(model.parameters(), hyperparams);
                Loss loss_fn = new CrossEntropyLoss();

                Misc.train(model, dl[0], dl[1], optim, loss_fn, 50);
        }

}
