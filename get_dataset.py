import tensorflow as tf

def get_dataset(model_name, type="train"):
    if model_name == "SGLD_LR":
        x = []
        label = []
        with open("dataset/SGLD_LR/%s.csv" % type, "r") as f:
            for l in f.readlines():
                x1, x2, y = l.split(",")
                x.append([float(x1), float(x2)])
                label.append(float(y))
        dataset = tf.data.Dataset.from_tensor_slices(
                    {
                        "data": tf.convert_to_tensor(x, tf.float32),
                        "label": tf.convert_to_tensor(label, tf.float32)
                    }
                )
        return dataset, len(x)
    else:
        raise "Invalid parameter for hparams.model"
