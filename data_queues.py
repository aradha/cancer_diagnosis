import numpy as np
import utils


# Data queues for loading in fair batches
class QueueManager():

    def __init__(self, train_data, train_labels):
        self.queues = {}
        for idx in range(len(train_labels)):
            label = np.argmax(train_labels[idx])
            dense_label = train_labels[idx]
            image = train_data[idx]
            if label in self.queues:
                self.queues[label][0].append((image, dense_label))
            else:
                self.queues[label] = [[(image, dense_label)], [], False]

    def draw_fair_batch(self, batch_size):
        sample_batch_size = int(batch_size / len(self.queues))
        batch_data = []
        batch_labels = []

        for label in self.queues:
            idx = 0
            in_queue = self.queues[label][0]
            out_queue = self.queues[label][1]

            while idx < sample_batch_size:

                if len(in_queue) > 0:
                    image, dense_label = in_queue.pop(0)
                    batch_data.append(image)
                    batch_labels.append(dense_label)
                    out_queue.append((image, dense_label))
                    idx += 1
                else:
                    permuted = list(np.random.permutation(out_queue))
                    while len(permuted) > 0:
                        in_queue.append(permuted.pop(0))
                        out_queue.pop(0)
                    self.queues[label][2] = True  # Indicates all data seen

        return np.array(batch_data), np.array(batch_labels)

    def all_data_seen(self):
        for label in self.queues:
            if self.queues[label][2] == False:
                return False
        return True

    def reset_seen_flags(self):
        for label in self.queues:
            self.queues[label][2] = False
        for label in self.queues:
            in_queue = self.queues[label][0]
            out_queue = self.queues[label][1]
            while len(out_queue) > 0:
                in_queue.append(out_queue.pop(0))
            in_queue = list(np.random.permutation(in_queue))
