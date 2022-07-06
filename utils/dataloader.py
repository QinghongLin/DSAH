import numpy as np

# The batch data block is produced for each iteration
def data_iterator(img224, batch_size):
    while True:
        idxs = np.arange(0, len(img224))
        np.random.shuffle(idxs)
        for batch_idx in range(0, len(img224), batch_size):
            cur_idxs = idxs[batch_idx:batch_idx + batch_size]
            image_batch = img224[cur_idxs]
            if len(image_batch) < batch_size:
                break
            image_batch = image_batch.astype("float32")

            yield image_batch, cur_idxs