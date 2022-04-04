import torch

def solve_squares(B, A):
    # 1. Perform QR decomposition of matrix A
    print("A", A.shape)
    print("B", B.shape)

    def prod(vec0, vec1):
        return (vec0 * vec1).sum()

    def norm(vec):
        return vec / (vec * vec).sum().sqrt()

    col0 = norm(A[:, 0])
    col1 = norm(A[:, 1] - prod(A[:, 1], col0) * col0)

    Q = torch.stack((col0, col1), axis=1)
    R = torch.tensor([[prod(A[:, 0], col0), prod(A[:, 1], col0)],
                      [0, prod(A[:, 1], col1)]])

    X = torch.matmul(torch.inverse(R), Q.transpose(1, 0))
    X = torch.matmul(X, B)
    return X

class LSTSQ(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input, A):
        return g.op("lstsq", input, A)

    @staticmethod
    def forward(self, input, A):
        return torch.lstsq(input, A)[0][:2]
