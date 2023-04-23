import com.nn.*;
import com.optim.*;
import com.utils.Misc;
import com.data.*;
import com.models.MLP;

import java.util.*;

public class SampleClassification {
    public static void main(String args[]) {
        Tensor[] data = Misc.generate_clusters(10, 1000, 4);
        Tensor x_train = data[0];
        Tensor y_train = data[1];

        // NNModule seq = new Sequential();
        // seq.add_module((NNModule) new Linear(4, 8, true));
        // seq.add_module((NNModule) new Sigmoid());
        // seq.add_module((NNModule) new Linear(8, 10, true));
        NNModule seq = new MLP(4, 8, 10);

        Loss loss_fn = new CrossEntropyLoss();
        Map<String, Float> optim_params = new HashMap<String, Float>();
        optim_params.put("lr", 0.1f);
        Optimizer optim = new SGD(seq.parameters(), optim_params);

        for (int epoch = 0; epoch < 1000; epoch++) {
            Tensor o = seq.forward(x_train);
            Tensor loss = loss_fn.criterion(o, y_train);
            if (epoch % 100 == 0) {
                System.out.println(epoch + ": " + loss);
            }
            optim.zero_grad();
            loss.backward();
            optim.step();
        }
    }
}
