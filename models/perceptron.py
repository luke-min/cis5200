import torch


class Perceptron:
    """
    Class to represent Perceptron algorithm.
    """
    def __init__(self, data):
        self.data = data

    def perceptron_margin(self, X, w_opt):
        # X := Tensor(float) of size (m,d) --- This is a batch of m examples
        #     of dimension d
        #
        # w_opt := Tensor(float) of size (d,) --- This is the ground truth linear
        #     weights
        #
        # Return := Tensor(float) of size (1,) --- This is the theoretical margin

        w_opt = torch.reshape(w_opt, (-1, 1))  # Reshape to column vector (d,1)

        return torch.min(torch.abs(torch.mm(X, w_opt))) / torch.linalg.vector_norm(w_opt)

    def perceptron_update_condition(self, xi, yi, w):
        # xi := Tensor(float) of size (d,) --- This is a single example
        #     of dimension d
        #
        # yi := Tensor(float) of size (1,) --- This is a single label for the
        #     example xi
        #
        # w := Tensor(float) of size (d,) --- This is the current estimate of the
        #     linear weights
        #
        # Return := Tensor(bool) of size (1,) --- This is true if the perceptron
        #     algorithm will do an update on this example, and false otherwise
        #
        return torch.sign(torch.dot(xi, w)) != torch.sign(yi)

    def perceptron_update_weight(self, xi, yi, w):
        # xi := Tensor(float) of size (d,) --- This is a single example of
        #     dimension d
        #
        # yi := Tensor(float) of size (1,) --- This is a single label for the
        #     example xi
        #
        # w := Tensor(float) of size (d,) --- This is the current estimate of the
        #     linear weights
        #
        # Return := Tensor(float) of size (d,) --- This is the updated linear
        #     weights after performing the perceptron update
        #
        return torch.add(w, xi, alpha=yi.item())

    def perceptron_algorithm(self, X, y, niter=100):
        m, d = X.size()
        w = torch.zeros(d)
        for t in range(niter):
            print('Iteration : ', t + 1)
            weight_updated = False
            for xi, yi in zip(X, y):
                if self.perceptron_update_condition(xi, yi, w):
                    w = self.perceptron_update_weight(xi, yi, w)
                    weight_updated = True
                    continue
            if not weight_updated:
                return w, t


