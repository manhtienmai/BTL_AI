import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        score = self.run(x)
        scalar_score = nn.as_scalar(score)
        return 1 if scalar_score >= 0 else -1
    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        while True:
            convergence = False
            for x, y in dataset.iterate_once(1):
                if self.get_prediction(x) != nn.as_scalar(y):
                    convergence = True
                    self.w.update(x, nn.as_scalar(y))
            if not convergence:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        #initialize weights and biases
        self.w1 = nn.Parameter(1, 512)
        self.b1 = nn.Parameter(1, 512)
        self.w2 = nn.Parameter(512, 1)
        self.b2 = nn.Parameter(1, 1)
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        x = nn.Linear(x, self.w1)
        x = nn.AddBias(x, self.b1)
        x = nn.ReLU(x)
        x2 = nn.AddBias(nn.Linear(x, self.w2), self.b2)
        return x2
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_value = self.run(x)
        return nn.SquareLoss(predicted_value, y)
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learning_rate = 0.05
        batch_size = 200

        while True:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                loss_scalar = nn.as_scalar(loss)
                if loss_scalar <= 0.02:
                    return
                gradients = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])
                self.w1.update(gradients[0], -learning_rate)
                self.b1.update(gradients[1], -learning_rate)
                self.w2.update(gradients[2], -learning_rate)
                self.b2.update(gradients[3], -learning_rate)
class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        input_size = 784
        hidden_layer_size = 200  # suggested size
        output_size = 10 # 10 classes (digits 0->9)

        self.W1 = nn.Parameter(input_size, hidden_layer_size)
        self.b1 = nn.Parameter(1, hidden_layer_size)
        self.W2 = nn.Parameter(hidden_layer_size, output_size)
        self.b2 = nn.Parameter(1, output_size)
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        x = nn.Linear(x, self.W1)
        x = nn.AddBias(x, self.b1)
        x = nn.ReLU(x)
        x = nn.Linear(x, self.W2)
        x = nn.AddBias(x, self.b2)
        return x
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_value = self.run(x)
        return nn.SoftmaxLoss(predicted_value, y)
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learning_rate = 0.5 # as suggested
        batch_size = 100 # suggest

        while True:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, [self.W1, self.b1, self.W2, self.b2])
                self.W1.update(gradients[0], -learning_rate)
                self.b1.update(gradients[1], -learning_rate)
                self.W2.update(gradients[2], -learning_rate)
                self.b2.update(gradients[3], -learning_rate)
            print(f"Accuracy: {dataset.get_validation_accuracy()}")
            if dataset.get_validation_accuracy() > 0.98:
                return

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        hidden_size = 200
        self.W_initial = nn.Parameter(self.num_chars, hidden_size)
        self.W_hidden = nn.Parameter(hidden_size, hidden_size)
        self.Wx = nn.Parameter(self.num_chars, hidden_size)
        self.W_final = nn.Parameter(hidden_size, len(self.languages))
        self.b_final = nn.Parameter(1, len(self.languages))
    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        trans = nn.Linear(xs[0], self.W_initial)

        # each character in word
        for c in xs[1:]:
            a = nn.Add(nn.Linear(c, self.Wx), nn.Linear(trans, self.W_hidden))
            trans = nn.ReLU(a)
        result = nn.AddBias(nn.Linear(trans, self.W_final), self.b_final)
        return result
    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_value = self.run(xs)
        return nn.SoftmaxLoss(predicted_value, y)
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        threshold = 0.87
        learning_rate = 0.1
        batch_size = 100

        while True:
            for xs, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(xs, y)
                gradients = nn.gradients(loss, [self.W_initial, self.W_hidden, self.Wx, self.W_final, self.b_final])

                #update value of parameters
                self.W_initial.update(gradients[0], -learning_rate)
                self.W_hidden.update(gradients[1], -learning_rate)
                self.Wx.update(gradients[2], -learning_rate)
                self.W_final.update(gradients[3], -learning_rate)
                self.b_final.update(gradients[4], -learning_rate)
            print(f"Accuracy: {dataset.get_validation_accuracy()}")
            if dataset.get_validation_accuracy() > threshold:
                return
