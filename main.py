import mnist
import random
from matplotlib import pyplot as plt
from typing import List

from tensor import Tensor
from tensor_interactions import shape, tensor_sum
from layer import Layer
from loss import Loss
from optimizer import Optimizer

train_images = mnist.train_images().tolist()
train_labels = mnist.train_labels().tolist()

assert shape(train_images) == [60000, 28, 28]
assert shape(train_labels) == [60000]

fig, ax = plt.subplots(10, 10);

for i in range(10):
    for j in range(10):
        ax[i][j].imshow(train_images[50000 + 10 * i + j], cmap="Greys")
        ax[i][j].xaxis.set_visible(False)
        ax[i][j].yaxis.set_visible(False)

plt.show()

test_images = mnist.test_images().tolist()
test_labels = mnist.test_labels().tolist()

assert shape(test_images) == [10000, 28, 28]
assert shape(test_labels) == [10000]

avg = tensor_sum(train_images) / 60000 / 28 / 28

train_images = [[(pixel - avg) / 256 for row in image for pixel in row]
                for image in train_images]

test_images = [[(pixel - avg) / 256 for row in image for pixel in row]
                for image in test_images]

assert shape(train_images) == [60000, 784]
assert shape(test_images) == [10000, 784]
assert -0.0001 < tensor_sum(train_images) < 0.0001

def one_hot_encode (i: int, num_labels: int = 10) -> List[float]:
    return [1.0 if j == i else 0.0 for j in range(num_labels)]

assert one_hot_encode(3) == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

train_labels = [one_hot_encode(label) for label in train_labels]
test_labels = [one_hot_encode(label) for label in test_labels]

assert shape(train_labels) == [60000, 10]
assert shape(test_labels) == [10000, 10]

def argmax(xs: List) -> int:
    return max(range(len(xs)), key=lambda i: xs[i])

import tqdm

def loop(model: Layer,
         images: List[Tensor],
         labels: List[Tensor],
         loss: Loss,
         optimizer: Optimizer = None) -> None:
    
    correct = 0
    total_loss = 0

    with tqdm.trange(len(images)) as t:
        for i in t:
            predicted = model.forward(images[i])
            if argmax(predicted) == argmax(labels[i]):
                correct += 1
            total_loss += loss.loss(predicted, labels[i])

            if optimizer is not None:
                gradient = loss.gradient(predicted, labels[i])
                model.backward(gradient)
                optimizer.step(model)

            avg_loss = total_loss / (i + 1)
            acc = correct / (i + 1)
            t.set_description(f"р-е рукопис. цифр, потеря: {avg_loss:.3f}, точн.: {acc:.3f}")


from linear_layer import Linear
from sse_loss import SSE
from momentum import Momentum

random.seed(0)

model = Linear(784, 10)
loss = SSE()
optimizer = Momentum(learning_rate=0.01, momentum=0.99)

loop(model, train_images, train_labels, loss, optimizer)